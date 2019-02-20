dt="$(date +"%l%P%d%b%y")"  #e.g. 9pm15May16
NOW=`echo ${dt//[[:blank:]]/}`
fout="/xhome/aoflagger/logs/log_$NOW.txt"

start=$(date -u +%s.%N)
c=0
for f in $@	 
do
    echo $f
    echo "processing file $f"
    time aoflagger -indirect-read -strategy /xhome/aoflagger/kat7-aoflagger-matrix.rfis  $f >> $fout
    let c=c+1
    
done

new=$(date -u +%s.%N)
elapsed="$(bc -l <<<"$new-$start")"
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
