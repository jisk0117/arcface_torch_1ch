set -x

image_path="/home/mkfr/src/train_tmp/IJBC"
result_dir="test_dirs"
target="IJBC"

model_prefixes=(
	"wf44m_mbf_20221212"
	)
#network="anynet"
networks=(
	"mbf"
	)
#params=(
#	"0197"
#	"0198"
#	"0199"
#	)

#python -u eval_ijbc.py --model-prefix $model_prefix --image-path $image_path --result-dir $result_dir --batch-size 32 --network $network --param $param --target $target > $job.log 2>&1 &
#python -u eval_ijbc.py --model-prefix $model_prefix --image-path $image_path --result-dir $result_dir --batch-size 512 --network $network --param $param --target $target

i=0
for model_prefix in ${model_prefixes[@]}
do
#	param=${params[$i]}
	network=${networks[$i]}
	log_path=$result_dir"/"$model_prefix"_"$target".log"
	i=$i+1

#	python -u eval_ijbc.py --model-prefix $model_prefix --image-path $image_path --result-dir $result_dir --batch-size 512 --network $network --param $param --target $target > $log_path
	python -u eval_ijbc.py --model-prefix $model_prefix --image-path $image_path --result-dir $result_dir --batch-size 512 --network $network --target $target > $log_path
done
