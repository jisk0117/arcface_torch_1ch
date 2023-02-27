set -x

image_path="/home/src/train_tmp/IJBC"
result_dir="test_dirs"
target="IJBC"

model_prefixes=(
#	"glint360k_anynet_curricular_0196_AnyNetXf_200_SE_ss"
#	"glint360k_anynet_curricular_0196_AnyNetXf_200_SE_rr_rank_glint360k_r100se_curricular"
#	"glint360k_anynet_curricular_0197_AnyNetXf_200_SE"
#	"glint360k_anynet_curricular_0197_AnyNetXf_200_SE_rr_rank_glint360k_r100se_curricular"
#	"glint360k_anynet_curricular_0198_AnyNetXf_200_SE"
#	"glint360k_anynet_curricular_0198_AnyNetXf_200_SE_rr_rank_glint360k_r100se_curricular"
#	"glint360k_anynet_curricular_0199_AnyNetXf_200_SE"
#	"glint360k_anynet_curricular_0199_AnyNetXf_200_SE_sr_rank_glint360k_r100se_curricular"
#	"glint360k_anynet_curricular_0199_AnyNetXf_200_SE_rr_rank_glint360k_r100se_curricular"
#	"glint360k_anynet_curricular_0200_AnyNetXf_200_SE_ss"
#	"glint360k_anynet_curricular_0200_AnyNetXf_200_SE_rr_rank_glint360k_r100se_curricular"
#	"wf44m_anynet_curricularface_0199_AnyNetXf_200_SE_sr_rank_wf44m_r100se_curricular"
#	"wf44m_anynet_curricularface_0199_AnyNetXf_200_SE_rr_rank_wf44m_r100se_curricular"
	"wf44m_anynet_curricularface_1024_0199_AnyNetXf_200_SE_rr_rank_wf44m_r100se_curricular"
	)
network="anynet"
params=(
#	"0196"
#	"0197"
#	"0198"
#	"0199"
	"0199"
#	"0200"
	)

#python -u eval_ijbc.py --model-prefix $model_prefix --image-path $image_path --result-dir $result_dir --batch-size 32 --network $network --param $param --target $target > $job.log 2>&1 &
#python -u eval_ijbc.py --model-prefix $model_prefix --image-path $image_path --result-dir $result_dir --batch-size 512 --network $network --param $param --target $target

i=0
for model_prefix in ${model_prefixes[@]}
do
	param=${params[$i]}
	log_dir=$result_dir"/"$model_prefix
	mkdir -p $log_dir

	for j in {0..19}
	do
		if [ "${#j}" -eq "1" ]; then
			model_num=0$j
		else
			model_num=$j
		fi
		
		log_path=$log_dir"/"$model_prefix"_"$model_num".log"
		python -u eval_ijbc_each.py --model-prefix $model_prefix --model-num $model_num --image-path $image_path --result-dir $result_dir --batch-size 4096 --network $network --param $param --target $target > $log_path
	done

	i=$i+1
done
