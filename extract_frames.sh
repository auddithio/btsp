OUT=/scr/aunag/ek100_frames
mkdir -p $OUT

find /vision/group/EPIC-KITCHENS-100 -path "*/rgb_frames/*.tar" | \
  parallel -j 64 'vid=$(basename {} .tar); \
    pid=$(echo $vid | cut -d_ -f1); \
    mkdir -p '"$OUT"'/$pid/$vid; \
    tar --skip-old-files -xf {} -C '"$OUT"'/$pid/$vid'

echo "Done extracting frames to $OUT"