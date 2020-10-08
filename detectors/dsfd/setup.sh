o "Downloading pretrained model"
gfileid="1ZVzJqbjoymnKl11jDc-VGkVgzBqR3rZZ"
destination_dir="./weights/"
file_name="dsfd_vgg_0.880.pth"
destination_path="${destination_dir}${file_name}"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${gfileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${gfileid}" -o ${destination_path}
