tmp_file="./tmp.txt"
failed_file="./failed.txt"
result="Success!!"

echo "" > "$failed_file"
for size in `seq 1 256`
do
  eval "./myexp -s ${size}" > "$tmp_file"
  if `grep -q "ClampedExp Failed!!!" "$tmp_file"`
  then
    echo "./myexp -s ${size}" >> "$failed_file"
    result="Failed!!"
  fi
done

rm "$tmp_file"
echo "$result"
