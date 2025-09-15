DATAPATH=/data/datasets #replace with your path
CODEPATH=/data/LLaVA #replace with your path


#export HF_ENDPOINT=https://hf-mirror.com # for China users


#GQA
cd $DATAPATH
mkdir gqa && cd gqa
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip

##or hf third part mirror, we found it faster but the size does not match 100%
#huggingface-cli download ej2/llava_mix665 \
#  --repo-type dataset \
#  --include "gqa.tar" \
#  --local-dir ./ #gqa, textvqa and coco2017
#
#tar -xvf gqa.tar
#rm gqa.tar


#OCR-VQA
#we provide third party mirror for faster download
huggingface-cli download  ej2/llava-ocr-vqa \
  --repo-type dataset \
  --local-dir ./ #llava-ocr-vqa
tar -xvf ocr_vqa.tar
rm ocr_vqa.tar

#there are a few corrupted images ocr-vqa, you can use the following script to remove them
cp $CODEPATH/playground/ocr-vqa/images_padding/1437717772.jpg $DATAPATH/ocr_vqa/images/1437717772.jpg
