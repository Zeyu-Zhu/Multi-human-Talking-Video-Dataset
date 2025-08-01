from models.audio_module import MultiHumanAudioBlock
import torch


if __name__ == '__main__':
    model = MultiHumanAudioBlock(in_channels=320, out_channels=320, feature_shape=(48, 64))
    
    audio_embeding = torch.zeros((1, 25, 12, 768))
    activate_score = [torch.zeros((1, 25, 1, 1)), torch.zeros((1, 25, 1, 1))]
    hidden_states = torch.zeros((1, 320, 25, 48, 64))
    
    face_mask = [[torch.zeros((1, 1, 25, 48, 64)).reshape(1, 25, 48*64)], [torch.zeros((1, 1, 25, 48, 64)).reshape(1, 25, 48*64)]]
    lip_mask = [[torch.zeros((1, 1, 25, 48, 64)).reshape(1, 25, 48*64)], [torch.zeros((1, 1, 25, 48, 64)).reshape(1, 25, 48*64)]]
    hidden_states = model.forward(hidden_states, 
                                  lip_mask_list=lip_mask, 
                                  face_mask_list=face_mask,
                                  audio_embedding=audio_embeding, 
                                  activate_score_list=activate_score)
    print(hidden_states.shape)