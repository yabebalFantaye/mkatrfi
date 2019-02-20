dt="$(date +"%l%P%d%b%y")"  #e.g. 9pm15May16
NOW=`echo ${dt//[[:blank:]]/}`
fout="/xhome/aoflagger/logs/log_$NOW.txt"

sfile="/xhome/KAT7/kat7-aoflagger-matrix.rfis"
#sfile="/xhome/KAT7/kat7-mild-flag.rfis"
echo "strategy file .."
cat  $sfile

startO=$(date -u +%s.%N)
c=0

logfile="/xhome/KAT7/log_latest_run.txt"
echo "$NOW" > $logfile
echo "" >> $logfile

TIMEFORMAT="%E";

fin=`echo $1`
d=`dirname $fin`
b=`basename $fin`
f=/data/$b
echo "$d $b $f"
echo $@

#for f in $@	 

for i in `seq 1 1`;
do
    rm -rf ${f} 

    echo "$i copying $fin..."
    cp -r ${fin} /data/ 
    echo "ls $f: "
    ls $f

    v=`du -csh $f | grep $f` #>> $logfile 
    echo "$i processing file $f"
    start=$(date -u +%s.%N)
    #unmasked flops count
    /xhome/sde2018/sde64 -knl -iform 1 -omix /xhome/KAT7/flops/flops${i}_knl_unmsk.out -top_blocks 5000 -- aoflagger \
	-indirect-read -strategy  $sfile $f 

    /xhome/sde2018/sde64 -knl -iform 1 -odyn_mask_profile /xhome/KAT7/flops/flops${i}_knl_msk.out -top_blocks 5000 -- aoflagger \
	-indirect-read -strategy  $sfile $f 
	#>> $fout
    new=$(date -u +%s.%N) 
    elapsed="$(bc -l <<<"$new-$start")"
    echo "time (in sec): ${elapsed}; file:  $v" >> $logfile
    #time aoflagger -indirect-read -strategy /xhome/KAT7/kat7-mild-flag.rfis  $f >> $fout
    let c=c+1

done

#copy processed file 
#echo "Loop done, copying processed file to $fin"
#rm -rf $fin
#mv $f $d/

new=$(date -u +%s.%N)
elapsed="$(bc -l <<<"$new-$startO")"
elapsed_single="$(bc -l <<<"$elapsed/$c")"
#elapsed=$((new - start/1000))
#elapsed_single=$((elapsed / c))
echo "--------------------"
echo "    Elapsed time for processing $c files:  $elapsed"
echo "    Elapsed time per file: $elapsed_single"
echo "--------------------"

cat >> $fout <<EOF

-------------
    Elapsed time for processing $c files:  $elapsed"
    Elapsed time per file: $elapsed_single"
------------- 

EOF
