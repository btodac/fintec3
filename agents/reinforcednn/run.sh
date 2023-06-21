#!/bin/bash
for j in {0..20};
do
    sed -i "s/restart = True/restart = False/g" main.py
    for i in {0..10}; 
    do 
        python main.py
        if [[ $? != 101 ]];
        then
            sed -i "s/restart = False/restart = True/g" main.py
        else
            break
        fi
    done
done
