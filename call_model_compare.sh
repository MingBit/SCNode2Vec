
#!/bin/bash

dirname=/home/mwu/MING_V9T/PhD_Pro/Test/Simulation/Node2Vec_BoolODE_Res/
for file in $dirname/G30*
do
    for drop in 0.25 0.5 0.75
      do
        filename=$(echo `basename "$file"`)
	if [[ $filename == *"PCA"* ]]; then
	break
	fi
        echo $filename':'$drop
        python /home/mwu/MING_V9T/PhD_Pro/SCNode2Vec/Test.py $filename 0.1 1.0 $drop
        python /home/mwu/MING_V9T/PhD_Pro/SCNode2Vec/Test.py $filename 1.0 0.1 $drop
      done
done
