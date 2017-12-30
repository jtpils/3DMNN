#!/bin/bash

#Get every single file
#Run it through the meshlab scripts which export it to the concept_processed folder

objects=$(ls ../../../data/concept)
here=$pwd

cd ../../../data/concept
script=../../main/utils/mesh_processing/catmull_subdivision.mlx
i=0
for object in $objects; do
    input=$object
    output=../concept_processed/$object
    meshlabserver -i $input -o $output -s $script &> /dev/null
    let i+=1
    echo "$i"
done
cd $here

echo "Done."
