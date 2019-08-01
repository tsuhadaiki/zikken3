#!/bin/zsh
name=$1
cnt=1
while read line
do
  curl $line > image1/$name$cnt.jpg
  cnt=`expr $cnt + 1`
done < $name".txt"