#!/bin/bash

outputFolder=$1
#fileCnt=$2

echo $outputFolder
#echo $fileCnt

mkdir -p $outputFolder

for dir in ./ishiharaPlates/*
do
	dirName=${dir##*/}
	mkdir -p "$outputFolder/$dirName"

	ls "$dir" | #sort -R | #tail -$fileCnt |
		while read file; do
			cp "$dir/$file" "$outputFolder/$dirName"			
		done
done
