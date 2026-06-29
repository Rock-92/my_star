def get_model(modelName):
    if modelName.lower() == 'cdnnet':
        from model.model_CDNNet import Res_CBAM_block, CDNNet_Det
        return Res_CBAM_block, CDNNet_Det
    elif modelName.lower() == 'dnanet':
        from model.model_DNANet import Res_CBAM_block, DNANet
        return Res_CBAM_block, DNANet
    elif modelName.lower() == 'unet':
        from model.model_UNet import Res_CBAM_block, UNet
        return Res_CBAM_block, UNet
    elif modelName.lower() == 'dnrfanet':
        from model.model_DNRFANet import ARFAM, DNRFANet
        return ARFAM, DNRFANet
    elif modelName.lower() == 'agpcnet':
        from model.model_AGPCNet import AGPCNet
        return AGPCNet
    elif modelName.lower() == 'hcfnet':
        from model.model_HCFNet import HCFNet_Det
        return HCFNet_Det
    elif modelName.lower() == 'serankdet':
        from model.model_SeRankDet import SeRankDet
        return SeRankDet

