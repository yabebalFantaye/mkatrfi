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

for f in $@	 
do
    v=`du -csh $f | grep $f` #>> $logfile 
    echo "processing file $f"
    start=$(date -u +%s.%N)
    time aoflagger -indirect-read -strategy  $sfile $f >> $fout
    new=$(date -u +%s.%N) 
    elapsed="$(bc -l <<<"$new-$start")"
    echo "time (in sec): ${elapsed}; file:  $v" >> $logfile
    #time aoflagger -indirect-read -strategy /xhome/KAT7/kat7-mild-flag.rfis  $f >> $fout
    let c=c+1
    
done

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
