if [ -d './qualitycheck/images/' ]; then
    echo "qualitycheck images already dl"
else
    echo "downloading qualitycheck images"
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1mHsmpzYhvyEX0bKIc3iL4-ReZe6gry4N' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1mHsmpzYhvyEX0bKIc3iL4-ReZe6gry4N" -O ./qualitycheck/images.tar.gz && rm -rf /tmp/cookies.txtOD
    tar -xvf ./qualitycheck/images.tar.gz -C ./qualitycheck/
    rm ./qualitycheck/images.tar.gz
    mv qualitycheck/visiomel_fixedres ./qualitycheck/images
fi

if [ -f './models/gigassl.pth.tar' ]; then
    echo "gigassl model already dl"
else
    echo "downloading gigassl model"
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1jnpmmeU8srkpOCRCiL8yXSHu9rnH0870' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1jnpmmeU8srkpOCRCiL8yXSHu9rnH0870" -O ./models/gigassl.pth.tar && rm -rf /tmp/cookies.txtOD
fi

if [ -f './models/moco.pth.tar' ]; then
    echo "moco model already dl"
else
    echo "downloading moco model"
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?              export=download&id=12vvx9jS01CWMGH5XIID48nJ4RKnVl01W' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12vvx9jS01CWMGH5XIID48nJ4RKnVl01W" -O ./models/moco.pth.tar && rm -rf /tmp/cookies.txtOD
fi
