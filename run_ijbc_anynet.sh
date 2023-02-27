set -x

image_path="/home/datasets/ijb/IJBC"; target="IJBC"
#model_path="test/search_04"
model_path="test/compared_04"
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
#	"wf44m_anynet_curricularface_1024_0199_AnyNetXf_200_SE_rr_rank_wf44m_r100se_curricular"
#	"wf44m_anynet_curricularface_0204_AnyNetXf_light_300_2.6_gw_sr_rank_wf44m_r100se_curricular"
#	"wf44m_anynet_curricularface_0204_AnyNetXf_light_300_2.6_gw_rr_rank_wf44m_r100se_curricular"
#	"wf44m_anynet_curricularface_0289_AnyNetXf_light_300_2.6_gw_sr_rank_wf44m_r100se_curricular"
#	"wf44m_anynet_curricularface_0289_AnyNetXf_light_300_2.6_gw_rr_rank_wf44m_r100se_curricular"
#	"wf44m_anynet_curricularface_0012_AnyNetXf_light_300_2.6_gw_sr_rank_wf44m_r100se_curricular"
#	"wf44m_anynet_curricularface_0012_AnyNetXf_light_300_2.6_gw_rr_rank_wf44m_r100se_curricular"
#	"wf44m_anynet_curricularface_0164_AnyNetXf_light_300_2.6_gw_sr_rank_wf44m_r100se_curricular"
#	"wf44m_anynet_curricularface_0164_AnyNetXf_light_300_2.6_gw_rr_rank_wf44m_r100se_curricular"
#	"wf44m_anynet_curricularface_0298_AnyNetXf_light_300_2.6_gw_sr_rank_wf44m_r100se_curricular"
#	"wf44m_anynet_curricularface_0298_AnyNetXf_light_300_2.6_gw_rr_rank_wf44m_r100se_curricular"
#	"wf44m_anynet_curricularface_0105_AnyNetXf_light_300_2.6_gw_sr_rank_wf44m_r100se_curricular"
#	"wf44m_anynet_curricularface_0105_AnyNetXf_light_300_2.6_gw_rr_rank_wf44m_r100se_curricular"
#	"wf44m_anynet_curricularface_0199_AnyNetXf_200_SE_sr_rank_wf44m_r100se_curricular"
	"wf44m_anynet_curricularface_0199_AnyNetXf_200_SE_rr_rank_wf44m_r100se_curricular"
	)
network="anynet"
format=(0000)
#params=(204 204 289 289 12 12 164 164 298 298 105)
#param_path="backbones_anynet/AnyNetXf_light_300_2.6_gw"
params=(199)
param_path="backbones_anynet/AnyNetXf_200_SE"

#result_dir="test_dirs"; mkdir -p $result_dir
result_dir=$model_path"/eval"; mkdir -p $result_dir

#python -u eval_ijbc.py --model-prefix $model_prefix --image-path $image_path --result-dir $result_dir --batch-size 32 --network $network --param $param --target $target > $job.log 2>&1 &
#python -u eval_ijbc.py --model-prefix $model_prefix --image-path $image_path --result-dir $result_dir --batch-size 512 --network $network --param $param --target $target

i=0
for model_prefix in ${model_prefixes[@]}
do
	idx=${params[$i]}
	param=${format:${#idx}:${#format}}${idx}
	log_path=$result_dir"/"$model_prefix".log"

	python -u eval_ijbc.py \
		--result-dir $result_dir --image-path $image_path --batch-size 8192 --target $target \
		--network $network --model-prefix $model_prefix --model-path $model_path \
	       	--param $param --param_path $param_path > $log_path
	i=$i+1
done
