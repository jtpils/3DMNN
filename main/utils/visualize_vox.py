import binvox_rw as bv

with open('/home/viktorv/Downloads/voxel.obj.binvox', 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)

print(model.data)

