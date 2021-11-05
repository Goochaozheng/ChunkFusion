# Training Data Generation

You can generate your own training data by following this [guide](),  
or download our training data from [here](link_to_chunk_h5)



## Step of Generating Chunk Data

A guide of generating own training data.  
A sample of data preprocessing can be found at `preprocess/sample`.

### Generate SDF Volume:  

```
python generate_SDF.py \
    --data_list_dir=./sample \
    --chunk_size=16 \
    --voxel_resolution=0.01
```

First scale the mesh model into actual size, to mimic real-world data;   
Then generate SDF volume from scaled mesh models using [mesh_to_sdf](https://github.com/marian42/mesh_to_sdf). This  
SDF values are in meter, not truncated.  

`data_list_dir`: path to model list and corresponding scale list  
- model_list: list containing path to the mesh models  
- scale_list: corresponding scale of the mesh model;  
The value in list is the inverse of scale, `scaled_size = origin_size / values`   

`chunk_size`: Number of voxels in a chunk.  
`voxel_resolution`: Resolution of voxels, in meter.

Processed model
The SDF voxel volume will be stored in numpy array and saved as .npy file.  
Corresponding mesh generated from SDF are also saved.

### Generate Depth Maps

```
python create_depth_scan.py \
    --data_dir=./sample \
    --scan_count=200 --scan_width=640 --scan_height=480 \
    --depth_min=0.1 --depth_max=1.0 \
    --focal_length=500 \
    --baseline_m=0.075
```

Generate random depth maps from mesh models.  
Kinect-like noise will be added to the depth maps with [simkinect](https://github.com/ankurhanda/simkinect).  
Intrinsics will be written to `intrinsic.txt`.

`data_dir`: path to the folder of saved mesh model.  
`scan_count`: Number of depth maps to be generated.  
`scan_width, scan_height`: Size of the generated depth mas;  
`depth_min, depth_max`: Min & max value of depth;  
`focal_length`: Focal length of camera;  
`baseline_m`: Parameter of kinect-like noise;  


