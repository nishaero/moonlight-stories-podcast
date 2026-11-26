import sys, torch, soundfile as sf, json
sys.argv=[sys.argv[0]]
from qwen_tts import Qwen3TTSModel
TXT={"de":("Hallo, mein Schatz. Na? Bist du schon ein bisschen müde? Schau mal, draußen ist es ganz dunkel, und die Sterne gucken zu uns herein. "
           "Pssst, hörst du das? Das ist der Mond, er singt ein leises Lied. Komm, kuschel dich ein. Ich erzähle dir eine ganz besondere Geschichte."),
     "en":("Hello, my little one. Hmm? Are you feeling a little sleepy? Look, it is all dark outside, and the stars are peeking in at us. "
           "Shh, can you hear that? That is the moon, singing a quiet little song. Come, snuggle in. I am going to tell you a very special story.")}
D={
 "F1":"A soft, warm, intimate female voice, about 30, a bedtime storyteller speaking very close to the microphone to a sleepy three-year-old: gentle sing-song child-directed speech with lots of melody, pitch rising and falling widely, tender and smiling, breathy and whispery in the quiet moments, slow and unhurried, playful and loving, never formal, never like an adult newsreader.",
 "F2":"A gentle young woman in her mid-20s, like a kind nursery storyteller from a children's podcast, with a light, sweet, slightly higher voice. She talks to a small child on her lap: very melodic, expressive, wondering on questions, whispering on secrets, a smile in every sentence, slow, soft and cosy, not an adult narrator.",
 "M1":"A soft, warm, gentle male voice, about 35, a loving dad telling a bedtime story to his sleepy three-year-old, close to the microphone: calm and cosy, melodic child-directed speech with gentle ups and downs, a smile in the voice, slow, whispery in the quiet moments, never formal, never like an announcer.",
 "M2":"A kind, soft-spoken man in his 30s with a warm, mellow baritone, like a friendly storyteller from a children's podcast. He speaks slowly and tenderly to a small child, with lively but quiet expressiveness, wondering on questions, hushed on secrets, smiling, natural, not an adult narrator.",
}
HESS=""
m=Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",device_map="cuda:0",dtype=torch.bfloat16)
for k,d in D.items():
    for lang in ("de","en"):
        for seed in (1,2):
            torch.manual_seed(seed)
            w,sr=m.generate_voice_design(text=TXT[lang],instruct=d+(HESS if lang=="de" else ""),language="German" if lang=="de" else "English",temperature=1.0,top_k=60,top_p=0.97)
            sf.write(f"voices/cand/{k}_{lang}_s{seed}.wav",w[0],sr); print(k,lang,seed,round(len(w[0])/sr),"s",flush=True)
json.dump({"text":TXT,"descs":D,"hess":HESS},open("voices/cand/cand.json","w"),ensure_ascii=False,indent=1)
