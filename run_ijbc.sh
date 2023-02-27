set -x

#image_path="/home/mkfr/src/train_tmp/IJBB"
image_path="/home/mkfr/src/train_tmp/IJBB"
result_dir="test_dirs"
#target="IJBB"
target="IJBB"

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
#	"wf42m_vit_b_221007"
#	"wf44m_mbf_221211"
	"wf44m_vit_b_221219"
	)
#network="anynet"
network="vit_b_dp005_mask_005"
params=(
#	"0196"
#	"0197"
#	"0198"
#	"0199"
	"0200"
	)

#python -u eval_ijbc.py --model-prefix $model_prefix --image-path $image_path --result-dir $result_dir --batch-size 32 --network $network --param $param --target $target > $job.log 2>&1 &
#python -u eval_ijbc.py --model-prefix $model_prefix --image-path $image_path --result-dir $result_dir --batch-size 512 --network $network --param $param --target $target

i=0
for model_prefix in ${model_prefixes[@]}
do
	param=${params[$i]}
	log_path=$result_dir"/"$model_prefix".log"
	i=$i+1

	python -u eval_ijbc.py --model-prefix $model_prefix --image-path $image_path --result-dir $result_dir --batch-size 512 --network $network --param $param --target $target > $log_path
done
