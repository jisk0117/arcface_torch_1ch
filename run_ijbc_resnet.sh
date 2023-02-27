set -x

image_path="/home/datasets/ijb/IJBC"; target="IJBC"
model_path="test/compared_04"
model_prefixes=(
#	"wf44m_mbf_curricular_ss"
#	"wf44m_r18se_curricular_ss"
#	"wf44m_r34se_curricular_ss"
	"wf44m_mbf_curricular_sr_rank_wf44m_r100se_curricular"
#	"wf44m_r18se_curricular_sr_rank_wf44m_r100se_curricular"
#	"wf44m_r34se_curricular_sr_rank_wf44m_r100se_curricular"
	)
networks=(
	"mbf"
#	"r18se"
#	"r34se"
	)
batch_sizes=(
	8192 
#	4096 
#	4096
	)
result_dir=$model_path"/eval"; mkdir -p $result_dir

#python -u eval_ijbc.py --model-prefix $model_prefix --image-path $image_path --result-dir $result_dir --batch-size 32 --network $network --param $param --target $target > $job.log 2>&1 &
#python -u eval_ijbc.py --model-prefix $model_prefix --image-path $image_path --result-dir $result_dir --batch-size 512 --network $network --param $param --target $target

i=0
for model_prefix in ${model_prefixes[@]}
do
	network=${networks[$i]}
	log_path=$result_dir"/"$model_prefix".log"
	batch_size=${batch_sizes[$i]}

#	python -u eval_ijbc.py --model-prefix $model_prefix --image-path $image_path --result-dir $result_dir --batch-size 512 --network $network --param $param --target $target > $log_path
	python -u eval_ijbc.py \
		--result-dir $result_dir --image-path $image_path --batch-size $batch_size --target $target \
		--network $network  --model-prefix $model_prefix --model-path $model_path > $log_path

	i=$i+1
done
