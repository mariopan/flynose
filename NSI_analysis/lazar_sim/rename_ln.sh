#!/bin/bash
# Basic while loop

for file in *.pickle
	do 
	mv $file ${file//lnspH/ln}
done

