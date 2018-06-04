for i in {0..100}
do
	if [ $i -lt 10 ];
	then 
		cp ./cocorealtrain2014/COCO_train2014_00000000000$i.jpg ./cocotrainsamples/;
	fi

	if [ $i -ge 10 -a $i -le 99 ];
	then
		cp ./cocorealtrain2014/COCO_train2014_0000000000$i.jpg ./cocotrainsamples/;
	fi

	if [ $i -gt 99 ];
	then
		cp ./cocorealtrain2014/COCO_train2014_000000000$i.jpg ./cocotrainsamples/;
	fi
done
