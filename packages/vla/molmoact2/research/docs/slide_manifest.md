# Slide 1
## Text
- MolmoAct2 on AMD Strix Halo
- Making it faster on local compute
- Identifying the compute bottlenecks
- ·
- Where can we be faster
- ·
- Smoother actions through blending
- ·
- Attention guided pruning
- Target:
- Smooth and real-time action policy on Ryzen AI MAX+ 39
- 5
## Refs
- (none)

# Slide 2
## Text
- What is MolmoAct2?
- Ai2's fully-open Vision-Language-Action 'action reasoning' model
- ▪  A VLA = a VLM backbone that perceives + reasons, grafted to a continuous action head.
- ▪  Molmo2-ER backbone: Qwen3-4B LLM + SigLIP2
- ViT
- ▪
- DiT
- flow-matching action expert: denoises future action chunks; conditioned on the VLM via per-layer KV cross-attention.
- ▪
- OpenFAST
- action tokenizer (2048 vocab) for the discrete/AR pretraining path; continuous flow path used at deployment.
- ▪  MolmoAct2-Think: optional adaptive depth-token reasoning (Depth-Anything-V2 + VQ-VAE 10x10x128).
- ▪
- Current model does not run on the
- strix
- -halo in real-time under bf16 quantization, how to make it faster?
- Inputs
- multi-cam RGB +
state[8] + language
- SigLIP2
- ViT
- + Connector
- pool 3rd/9th-last
→ MLP project
- Qwen3-4B VLM
- 36 layers · runs
ONCE / step (~550ms)
- DiT Flow Expert
- 36 layers · flow loop
~20 ms / step
- Action chunk
- (15,8) abs joint
+ gripper → robot
- per-layer KV: VLM layer-ℓ (K,V) → adapters Pₖ,Pᵥ → cross-attn of expert block ℓ
## Refs
- (none)

# Slide 3
## Text
- Milestones
- Shipped as
- MolmoAct2
- Ryzers
- package
- Milestone
- Description
- Status
- F
- ull-model smoke
- MolmoAct2-DROID load + 1 prediction on ROCm
- ✅
- DROID open-loop replay
- numeric correctness vs teleop GT (video+plot)
- ✅
- LIBERO closed-loop
- task success in MuJoCo sim (40-rollout sweep)
- ✅
- Interactive (synchronous
- - idealistic
- )
- live browser-driven chunkwise control
- ✅
- Real-time (async
- hronous - realistic
- )
- sim runs while planner thinks; hold-while-thinking
- ✅
- UR5e + XARM6 Cross Embodiment
- Trying to see if policy works across models
- ✅
- Has a
- ryzers
- Having it up on
- Github
- for sending to people
- ✅
- Post-
- ViT
- Random Token Pruning
- Sending less vision tokens to VLM makes it run faster
- ✅
- Action blending
- Running the model as soon as observations stream in
- ✅
- Pre-
- ViT
- Random token Pruning
- Trying to save compute on the vision encoder too
- ⚠️
- Attention guided token pruning
- Deterministically pruning vision patches/tokens
- ⚠️⚠️
## Refs
- (none)

# Slide 4
## Text
- Real-time demo (asynchronous)
- sim thread @20 Hz + async planner thread; arm HOLDS while thinking
- goal_0
- object_3
- spatial_9
- Task
- 
- Observation
- 
- Decision
- 
- Rollout
- 
- Hold
-  Done
- |________________________________________|
## Refs
- rId1 | http://schemas.microsoft.com/office/2007/relationships/media | ../media/media1.mp4
- rId8 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image1.png
- rId3 | http://schemas.microsoft.com/office/2007/relationships/media | ../media/media2.mp4
- rId9 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image2.png
- rId5 | http://schemas.microsoft.com/office/2007/relationships/media | ../media/media3.mp4
- rId10 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image3.png
- rId2 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/video | ../media/media1.mp4
- rId4 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/video | ../media/media2.mp4
- rId6 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/video | ../media/media3.mp4

# Slide 5
## Text
- Cross Embodiment Test
- Trying out MolmoAct2 Policy with UR5e and XARM6 robot arms
- Panda Franka
- XARM6
- UR5e
- Task
- 
- Observation
- 
- Decision
- 
- IK Transform
- 
- Rollout
-  Done
- |_______________________________________________|
## Refs
- rId1 | http://schemas.microsoft.com/office/2007/relationships/media | ../media/media4.mp4
- rId8 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image4.png
- rId3 | http://schemas.microsoft.com/office/2007/relationships/media | ../media/media5.mp4
- rId9 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image5.png
- rId5 | http://schemas.microsoft.com/office/2007/relationships/media | ../media/media6.mp4
- rId10 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image6.png
- rId2 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/video | ../media/media4.mp4
- rId4 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/video | ../media/media5.mp4
- rId6 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/video | ../media/media6.mp4

# Slide 6
## Text
- Targeting Efficiency
- How to make the model run faster and smoother on
- strix
- -halo
- Inputs
- multi-cam RGB +
state[8] + language
- SigLIP2
- ViT
- + Connector
- pool 3rd/9th-last
→ MLP project
- Qwen3-4B VLM
- 36 layers · runs
ONCE / step (~550ms)
- DiT Flow Expert
- 36 layers · flow loop
~20 ms / step
- Action chunk
- (15,8) abs joint
+ gripper → robot
- Blocks that consume compute
- One pass chunks out 15 action steps
- 238
- 284
- 79
- 601ms @bf16
## Refs
- rId3 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image7.png

# Slide 7
## Text
- Targeting Efficiency
- Why we need to make it faster and smoother
## Refs
- rId5 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image7.png
- rId1 | http://schemas.microsoft.com/office/2007/relationships/media | ../media/media7.mp4
- rId6 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image8.png
- rId2 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/video | ../media/media7.mp4

# Slide 8
## Text
- Token Pruning 1 – Post-
- ViT
- Random Drop
- Reducing the input token count to the VLM backbone
- Inputs
- multi-cam RGB +
state[8] + language
- SigLIP2
- ViT
- + Connector
- pool 3rd/9th-last
→ MLP project
- Qwen3-4B VLM
- 36 layers · runs
ONCE / step (~550ms)
- DiT Flow Expert
- 36 layers · flow loop
~20 ms / step
- Action chunk
- (15,8) abs joint
+ gripper → robot
- Prune the tokens here, keep x% tokens
## Refs
- rId3 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image9.png

# Slide 9
## Text
- Token Pruning 1 – Post-
- ViT
- Random Drop
- Reducing the input token count to the VLM backbone
- Inputs
- multi-cam RGB +
state[8] + language
- SigLIP2
- ViT
- + Connector
- pool 3rd/9th-last
→ MLP project
- Qwen3-4B VLM
- 36 layers · runs
ONCE / step (~550ms)
- DiT Flow Expert
- 36 layers · flow loop
~20 ms / step
- Action chunk
- (15,8) abs joint
+ gripper → robot
- Prune the tokens here, keep x% tokens
## Refs
- rId3 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image10.png

# Slide 10
## Text
- Token Pruning 2 – Pre-
- ViT
- Random Drop
- Reducing the input token count to the VLM backbone
- Inputs
- multi-cam RGB +
state[8] + language
- SigLIP2
- ViT
- + Connector
- pool 3rd/9th-last
→ MLP project
- Qwen3-4B VLM
- 36 layers · runs
ONCE / step (~550ms)
- DiT Flow Expert
- 36 layers · flow loop
~20 ms / step
- Action chunk
- (15,8) abs joint
+ gripper → robot
- Prune the patches here, keep x% patch groups
- 
- 50 percent is a good working point
## Refs
- rId3 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image11.png
- rId4 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image12.png

# Slide 11
## Text
- Action Blending
- Don’t wait for task to complete
- 
- start working as soon as observations are available
## Refs
- rId3 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image7.png

# Slide 12
## Text
- Action Blending
- Don’t wait for task to complete
- 
- start working as soon as observations are available
## Refs
- rId1 | http://schemas.microsoft.com/office/2007/relationships/media | ../media/media8.mp4
- rId5 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image13.png
- rId2 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/video | ../media/media8.mp4

# Slide 13
## Text
- Feedback?
- Don’t prune tokens randomly, prune based on feedback
- Transformer VLM blocks have a neat feature
-  we can observe attention scores between task
- Reference:
- https://github.com/CognitiveAISystems/BlindVLA
## Refs
- rId3 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image14.png

# Slide 14
## Text
- Feedback From Where?
- Action-Vision attention or Action-Text attention
- Vision-Text attention: Performs less
- Vision-Action attention: Performs better
## Refs
- rId3 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image15.png

# Slide 15
## Text
- Action-Attention Pruning Pre-
- ViT
- Keep only the important patches from action attention
- Inputs
- multi-cam RGB +
state[8] + language
- SigLIP2
- ViT
- + Connector
- pool 3rd/9th-last
→ MLP project
- Qwen3-4B VLM
- 36 layers · runs
ONCE / step (~550ms)
- DiT Flow Expert
- 36 layers · flow loop
~20 ms / step
- Action chunk
- (15,8) abs joint
+ gripper → robot
- Prune patches, keep x% patch groups
## Refs
- rId3 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image16.png

# Slide 16
## Text
- Action-Attention Pruning Post-
- ViT
- Keep only the important tokens from action attention
- Inputs
- multi-cam RGB +
state[8] + language
- SigLIP2
- ViT
- + Connector
- pool 3rd/9th-last
→ MLP project
- Qwen3-4B VLM
- 36 layers · runs
ONCE / step (~550ms)
- DiT Flow Expert
- 36 layers · flow loop
~20 ms / step
- Action chunk
- (15,8) abs joint
+ gripper → robot
- Prune tokens, keep x% tokens
## Refs
- rId3 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image17.png

# Slide 17
## Text
- What Does the Model See?
- A look into the pruning experiment
- 25%
- 
- tokens
- 
- 50%
- tokens
## Refs
- rId1 | http://schemas.microsoft.com/office/2007/relationships/media | ../media/media9.mp4
- rId7 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image18.png
- rId3 | http://schemas.microsoft.com/office/2007/relationships/media | ../media/media10.mp4
- rId8 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image19.png
- rId2 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/video | ../media/media9.mp4
- rId4 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/video | ../media/media10.mp4

# Slide 18
## Text
- Is It Useful?
- An unfortunate outcome – but without any sort of training so there is hope!
- Post-
- ViT
- guided pruning still underperforms compared to random pruning (Pre-
- ViT
- has it worse)
- We don’t have a real-time feedback mechanism yet
- Right now we are doing a dual pass through the VLM to find attention scores.
- Camera inputs change after taking action – feedback from previous state is sort of useless.
- We need a WAM style predictor to determine where to focus next.
## Refs
- rId3 | http://schemas.openxmlformats.org/officeDocument/2006/relationships/image | ../media/image20.png
