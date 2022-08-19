#!/bin/bash

# This is a yet another download script for the cmu arctic speech corpus.
# The corpus will be downloaded in $HOME/data/cmu_arctic/

# Source: https://gist.github.com/r9y9/ff67c05aeb87410eae2e

location=$HOME/data/cmu_arctic/

if [ ! -e $location ]
then
    echo "Create " $location
    mkdir -p $location
fi

root=http://festvox.org/cmu_arctic/packed/

cd $location

function download() {
    identifier=$1
    file=$2
    echo "start downloading $identifier, $file"
    mkdir -p tmp
    curl -L -o tmp/${identifier}.tar.bz2 $file
    tar xjvf tmp/${identifier}.tar.bz2
    rm -rf tmp
}

for f in aew ahw aup awb axb bdl clb eey fem gka jmk ksp ljm lnh rms rxr slp slt
do
    zipfile=${root}cmu_us_${f}_arctic.tar.bz2
    download $f $zipfile
done
