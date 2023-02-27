set -x

# base configuration ###############################################################################
#cfg=glint360k_mbf_1ch
cfgs=(
	wf44m_mbf_curricular
	wf44m_r18se_curricular
	wf44m_r34se_curricular
#	wf44m_mbf_1ch
#	wf44m_r34_cosface
#	wf44m_r18_cosface
)

# backbone resume configuration ####################################################################
#cfg_resumes=(
#	glint360k_anynet_curricular_0196_AnyNetXf_200_SE_ss
#	glint360k_anynet_curricular_0197_AnyNetXf_200_SE
#	glint360k_anynet_curricular_0198_AnyNetXf_200_SE
#	glint360k_anynet_curricular_0199_AnyNetXf_200_SE
#	glint360k_anynet_curricular_0200_AnyNetXf_200_SE_ss
#	wf44m_mbf_20221212
#	)
# ##################################################################################################

# rank configuration ###############################################################################
#cfg_ranks="configs/"rank_wf44m_mbf_20221212
# ##################################################################################################

for cfg in ${cfgs[@]}
do
	output="work_dirs/"$cfg
	if [[ ! -v cfg_resumes ]]; then
		output=$output"_s"
	        cfg_resume=None
	else
		output=$output"_r"
	#        cfg_resume="work_dirs/"${cfg_resumes[$i]}
		cfg_resume="work_dirs/"$cfg_resumes
	fi

	if [[ ! -v cfg_ranks ]]; then
		output=$output"s"
		cfg_rank=None
	else
		output=$output"r_"$cfg_ranks
		cfg_rank=${cfg_ranks[$i]}
	fi

	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
		--nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12345 train.py \
		configs/$cfg \
		--output=$output \
		--config_rank=$cfg_rank \
		--config_resume=$cfg_resume

	ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
done
