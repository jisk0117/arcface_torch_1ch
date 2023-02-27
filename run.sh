set -x

# base configuration ###############################################################################
#cfg=glint360k_r100se_arcface
#cfg=glint360k_r100se_curricular
#cfg=glint360k_r18se_curricular
#cfg=glint360k_r34se_curricular
#cfg=glint360k_anynet_curricular
#cfg=wf44m_anynet_cosface
cfg=wf44m_anynet_curricularface

param=(AnyNetXf_200_SE)
indexes=(
#	0196
#	0197
#	0198
	0199
#	0200
	)
# ##################################################################################################


# backbone resume configuration ####################################################################
#cfg_resumes=(
#	glint360k_anynet_curricular_0196_AnyNetXf_200_SE_ss
#	glint360k_anynet_curricular_0197_AnyNetXf_200_SE
#	glint360k_anynet_curricular_0198_AnyNetXf_200_SE
#	glint360k_anynet_curricular_0199_AnyNetXf_200_SE
#	glint360k_anynet_curricular_0200_AnyNetXf_200_SE_ss
#	)
# ##################################################################################################


# rank configuration ###############################################################################
#cfg_ranks=rank_glint360k_r100se_curricular
cfg_ranks=rank_wf44m_r100se_curricular
# ##################################################################################################


i=0
for idx in ${indexes[@]}
do
	output=$cfg"_"$idx"_"$param

	# resume configuration #####################################################################
	if [[ ! -v cfg_resumes ]]; then
		output=$output"_s"
		cfg_resume=None
	else
		output=$output"_r"
		cfg_resume=${cfg_resumes[$i]}
	fi

	# output path according to rank configuration  #############################################
	if [[ ! -v cfg_ranks ]]; then
		output=$output"s"
		cfg_rank=None
	else
		output=$output"r_"$cfg_ranks
		cfg_rank=$cfg_ranks
	fi
	
	i=$i+1
	# ##########################################################################################

	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
		--nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12345 train.py \
		configs/$cfg \
		--params=./backbones_anynet/$param/$idx.param \
		--output=work_dirs/$output \
		--config_rank=configs/$cfg_rank \
		--config_resume=work_dirs/$cfg_resume

	ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
done
