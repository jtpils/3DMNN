#!/bin/bash

#Get every single file
#Run it through the meshlab scripts which export it to the concept_processed folder

printf_new() {
 str=$1
 num=$2
 v=$(printf "%-${num}s" "$str")
 echo "${v// /*}"
}

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
     echo $i
 done

echo "Done applying algorithms."
sleep 1
echo "Extracting vertices..."

cd ../concept_processed
objects=$(ls)

# read -p "Press [Enter] key to start vertex extraction."

for object in $objects; do
    csplit -z $object /'7778 vertices, 0 vertices normals'/ {*} -f $object
done

echo "Deleting unnnesseccesary files..."

rm cube*.obj01
rm cube*.obj

echo "Done."
cd $here
