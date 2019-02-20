rootpath="/scratch2"
namekey="1375091767.full_pol"
rdir=${rootpath}/KAT7_data/KAT7_data/original/
for f in ${rdir}/listobs/${namekey}.ms;
do
fvalid=${rdir}/$(basename  $f)
echo "copying $fvalid to flagged folder"
cp -r $fvalid  ${rootpath}/KAT7_data/aoflagged/flagged/
done

#echo "copying file to flagged folder..."
#rm -r /data/KAT7_data/aoflagged/flagged/1375091767.full_pol.ms
#cp -r /data/KAT7_data/KAT7_data/original/1375091767.full_pol.ms  /data/KAT7_data/aoflagged/flagged/
echo "file copied. Running AOFlagger.."
echo ""

#aoflag 
nvidia-docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
-v /home/yabebal:/xhome \
-v ${rootpath}:/data \
--name ttx aoflagger/kern:latest \
/bin/bash /xhome/KAT7/run_aoflag_kat7.sh /data/KAT7_data/aoflagged/flagged/${namekey}.ms

#/data/KAT7_data/aoflagged/flagged/1416427899.full_pol.ms

date
echo "processed files are:"
du -csh ${rootpath}/KAT7_data/aoflagged/flagged/${namekey}.ms


