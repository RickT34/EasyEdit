from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from env import ModelEnv
import env
from tqdm import tqdm

def llm_generate(model_env: ModelEnv, d: int, prompts: list[str], batch_size: int = 8, max_newtok: int = 32):
    model_name = model_env.path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device(f"cuda:{d}" )
    model.to(device)

    tokenizer.pad_token = tokenizer.eos_token
    results = []
    
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        
        # 编码当前批次
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        
        # 生成文本（禁用采样以提高速度）
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_newtok,              # 最大新生成token数
                # stop_strings=[".", "\n", tokenizer.eos_token],
                tokenizer=tokenizer,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码并保存结果（跳过输入部分）
        generated = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        results+=generated
    
    return results

if __name__ == "__main__":
    model = env.ModelLLaMA3
    data = env.DatasetMQCFAllEdges.read()
    prompts = data[0]['tests'][:10]
    generated = llm_generate(model, 1, prompts)
    print(prompts, generated)