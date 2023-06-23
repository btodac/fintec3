#!/bin/bash

restart=$1
for j in {0..20};
do
    if [ $restart ]; then
        sed -i "s/restart = False/restart = True/g" main.py
    else
        sed -i "s/restart = True/restart = False/g" main.py
    fi
    for k in {0..10}; 
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
