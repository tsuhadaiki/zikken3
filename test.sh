curl "$1" > $2.html
python3 giul.py $2
sh getImage.sh $2