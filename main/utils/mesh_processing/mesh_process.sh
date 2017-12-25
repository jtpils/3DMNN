#!/bin/bash

#Get every single file
#Run it through the meshlab scripts which export it to the concept_processed folder

objects=$(ls ../../../data/concept)
here=$pwd

cd ../../../data/concept
for object in $objects; do
    input=$object
    output=../concept_processed/$object
    script=../../main/utils/mesh_processing/catmull_subdivision.mlx
    meshlabserver -i $input -o $output -s $script
done
cd $here

echo "Done."