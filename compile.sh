arch=soc

if [ -n "$1" ];then
	arch=$1
fi

DIR=build_$arch

rm -fr $DIR && mkdir $DIR
cd $DIR && cmake -DTARGET_ARCH=$arch ..
make -j4
cd ..



