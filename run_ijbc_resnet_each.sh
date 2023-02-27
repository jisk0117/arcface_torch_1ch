set -x

image_path="/home/src/train_tmp/IJBC"
result_dir="test_dirs"
target="IJBC"

model_prefixes=(
#	"glint360k_anynet_curricular_0197_AnyNetXf_200_SE"
#	"glint360k_anynet_curricular_0198_AnyNetXf_200_SE"
#	"glint360k_anynet_curricular_0199_AnyNetXf_200_SE"
#	"glint360k_r18se_curricular"
#	"glint360k_r34se_curricular"
#	"wf44m_r18se_curricularface"
#	"wf44m_r34se_curricularface"
	"wf44m_r50se_curricularface"
#	"wf44m_r100se_curricularface"
	)
networks=(
#	"r18se"
#	"r34se"
	"r50se"
#	"r100se"
	)

#python -u eval_ijbc.py --model-prefix $model_prefix --image-path $image_path --result-dir $result_dir --batch-size 32 --network $network --param $param --target $target > $job.log 2>&1 &
#python -u eval_ijbc.py --model-prefix $model_prefix --image-path $image_path --result-dir $result_dir --batch-size 512 --network $network --param $param --target $target

i=0
for model_prefix in ${model_prefixes[@]}
do
	network=${networks[$i]}
	log_dir=$result_dir"/"$model_prefix
	mkdir -p $log_dir

	for j in {17..19}
	do
		if [ "${#j}" -eq "1" ]; then
			model_num=0$j
		else
			model_num=$j
		fi

		log_path=$log_dir"/"$model_prefix"_"$model_num".log"
#		python -u eval_ijbc.py --model-prefix $model_prefix --image-path $image_path --result-dir $result_dir --batch-size 512 --network $network --param $param --target $target > $log_path
		python -u eval_ijbc_each.py --model-prefix $model_prefix --model-num $model_num --image-path $image_path --result-dir $result_dir --batch-size 2048 --network $network --target $target > $log_path
	done

	i=$i+1
done
