import numpy as np
from netCDF4 import Dataset
import sys

# -------- 用户可改的参数 --------
src_path = "nc/gdas/2018091100.nc"  # 源大 nc 文件路径（修改为你的文件）
dst_path = "upper_without.nc"          # 输出文件名
pressure_levels = ['1000', '925', '850', '700', '600', '500', '400', '300', '250', '200', '150', '100', '50']
# ---------------------------------

# mapping: output_var_name -> source variable name pattern (xxx 替换为层值)
var_patterns = {
    "z": "HGT:{lev} mb",   # HGT is geopotential height (m) in your source -> convert to geopotential (m2/s2)
    "t": "TMP:{lev} mb",
    "u": "UGRD:{lev} mb",
    "v": "VGRD:{lev} mb",
    "q": "SPFH:{lev} mb",
}

# physical constant
g = 9.80665  # m/s^2

# helper: try to find standard dimension names in source
def find_dim_name(ds, candidates):
    for c in candidates:
        if c in ds.dimensions:
            return c
    return None

def main():
    # sort pressure levels numeric descending (1000 -> 50)
    pressure_levels_sorted = sorted([int(p) for p in pressure_levels], reverse=True)
    pressure_levels_sorted = [str(p) for p in pressure_levels_sorted]  # strings like '1000','925',...
    nplevels = len(pressure_levels_sorted)

    src = Dataset(src_path, 'r')

    # detect likely dimension names (common alternatives)
    time_dim = find_dim_name(src, ('valid_time', 'time', 't', 'forecast_time'))
    lat_dim = find_dim_name(src, ('latitude', 'lat', 'y'))
    lon_dim = find_dim_name(src, ('longitude', 'lon', 'x'))

    if time_dim is None:
        src.close()
        raise RuntimeError("无法在源文件中找到时间维度（尝试了: valid_time/time/t/forecast_time）。请检查源文件。")
    if lat_dim is None or lon_dim is None:
        src.close()
        raise RuntimeError("无法在源文件中找到经纬度维度（尝试了常见名称）。请检查源文件。")

    # read coords
    src_lat = src.variables[lat_dim][:]
    src_lon = src.variables[lon_dim][:]
    ntime = src.dimensions[time_dim].size
    nlat = src_lat.size
    nlon = src_lon.size

    # prepare lat reversed (source is -90..90 -> we want 90..-90)
    # if source already descending (90..-90) keep as is; otherwise reverse.
    if src_lat[0] < src_lat[-1]:
        # increasing (e.g. -90..90) -> flip
        out_lat = src_lat[::-1].astype(np.float32)
        need_flip_lat = True
    else:
        out_lat = src_lat.astype(np.float32)
        need_flip_lat = False

    # prepare pressure_level numeric array (int)
    pressure_vals = np.array([int(p) for p in pressure_levels_sorted], dtype=np.int32)

    # create destination file
    dst = Dataset(dst_path, 'w', format='NETCDF4')
    # create dimensions
    dst.createDimension(time_dim, ntime)
    dst.createDimension('pressure_level', nplevels)
    dst.createDimension('latitude', nlat)
    dst.createDimension('longitude', nlon)

    # copy/create time variable if exists in src.variables
    if time_dim in src.variables:
        tvar_src = src.variables[time_dim]
        dst_t = dst.createVariable(time_dim, tvar_src.dtype, (time_dim,))
        try:
            dst_t[:] = tvar_src[:]
        except Exception:
            dst_t[:] = np.arange(ntime, dtype=np.int32)
        # copy attributes
        for att in getattr(tvar_src, "ncattrs", lambda: [])():
            try:
                dst_t.setncattr(att, getattr(tvar_src, att))
            except Exception:
                pass
    else:
        # create simple time index
        dst_t = dst.createVariable(time_dim, 'i4', (time_dim,))
        dst_t[:] = np.arange(ntime, dtype=np.int32)
        dst_t.long_name = "valid_time_index"

    # create pressure_level, lat, lon variables
    pvar = dst.createVariable('pressure_level', 'i4', ('pressure_level',))
    pvar[:] = pressure_vals
    pvar.long_name = "pressure_level_hPa"

    lat_var = dst.createVariable('latitude', 'f4', ('latitude',))
    lat_var[:] = out_lat
    try:
        lat_units = src.variables[lat_dim].units
    except Exception:
        lat_units = "degrees_north"
    lat_var.units = lat_units

    lon_var = dst.createVariable('longitude', 'f4', ('longitude',))
    lon_var[:] = src_lon.astype(np.float32)
    try:
        lon_units = src.variables[lon_dim].units
    except Exception:
        lon_units = "degrees_east"
    lon_var.units = lon_units

    # create data variables in dst
    dst_vars = {}
    for out_name in var_patterns.keys():
        v = dst.createVariable(out_name, 'f4', (time_dim, 'pressure_level', 'latitude', 'longitude'),
                               zlib=True, complevel=4, fill_value=np.float32(np.nan))
        # default units/long_name (will be overwritten below where possible)
        v.units = ""
        v.long_name = out_name
        dst_vars[out_name] = v

    # iterate pressure levels and copy data level-by-level to avoid爆内存
    for ilev, lev in enumerate(pressure_levels_sorted):
        print(f"处理层 {lev} hPa ...")
        # for each variable pattern
        for out_name, pattern in var_patterns.items():
            src_varname = pattern.format(lev=lev)
            if src_varname not in src.variables:
                src.close()
                dst.close()
                raise RuntimeError(f"在源文件中未找到变量: {src_varname}（期望用于输出字段 {out_name}）。请确认命名规则或源文件内容。")

            src_var = src.variables[src_varname]
            # read data for this level (可能返回 MaskedArray)
            data_raw = src_var[:]

            # get axes positions in src_var
            src_dims = src_var.dimensions  # tuple like ('valid_time','latitude','longitude')
            # find indices
            try:
                time_idx = src_dims.index(time_dim)
                lat_idx = src_dims.index(lat_dim)
                lon_idx = src_dims.index(lon_dim)
            except ValueError:
                # can't find required dims
                src.close()
                dst.close()
                raise RuntimeError(f"变量 {src_varname} 的维度中找不到 time/lat/lon（在 {src_dims} 中）。")

            # reorder to (time, lat, lon)
            perm = (time_idx, lat_idx, lon_idx)
            if perm != (0,1,2):
                tmp = data_raw
                # move time axis to 0
                if time_idx != 0:
                    tmp = np.moveaxis(tmp, time_idx, 0)
                    if lat_idx > time_idx:
                        lat_idx -= 1
                    if lon_idx > time_idx:
                        lon_idx -= 1
                # move lat to 1
                if lat_idx != 1:
                    tmp = np.moveaxis(tmp, lat_idx, 1)
                    if lon_idx > lat_idx:
                        lon_idx -= 1
                # move lon to 2
                if lon_idx != 2:
                    tmp = np.moveaxis(tmp, lon_idx, 2)
                data = tmp
            else:
                data = data_raw

            # convert masked values to nan, maintain float64 for safe multiply
            if isinstance(data, np.ma.MaskedArray):
                data = data.filled(np.nan).astype(np.float64)
            else:
                data = np.array(data, dtype=np.float64)

            # flip latitude if needed: after reorder lat is axis 1
            if need_flip_lat:
                data = data[:, ::-1, :]

            # final shape check
            if data.shape != (ntime, nlat, nlon):
                src.close()
                dst.close()
                raise RuntimeError(f"读入数据形状与期望不符: 变量 {src_varname} 得到形状 {data.shape}，期望 ({ntime},{nlat},{nlon})。")

            # apply geopotential conversion if needed
            if out_name == "z":
                # assume HGT_xxxmb is geopotential height in meters -> convert to geopotential (m^2/s^2)
                data = data * g
                # set units / long_name explicitly
                dst_vars[out_name].units = "m2 s-2"
                dst_vars[out_name].long_name = "geopotential"
            else:
                # try to copy units/long_name from source if present
                try:
                    if hasattr(src_var, 'units'):
                        dst_vars[out_name].units = getattr(src_var, 'units')
                except Exception:
                    pass
                try:
                    if hasattr(src_var, 'long_name'):
                        dst_vars[out_name].long_name = getattr(src_var, 'long_name')
                except Exception:
                    pass

            # ensure dtype float32 for writing
            data_out = data.astype(np.float32)

            # write into dst var at pressure level index ilev
            dst_vars[out_name][:, ilev, :, :] = data_out

    # optional: set global attributes
    dst.title = "upper-level fields extracted"
    dst.source = src_path

    src.close()
    dst.close()
    print("完成：已写入", dst_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("出错：", e)
        sys.exit(1)
