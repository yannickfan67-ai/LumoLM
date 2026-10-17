"""Build reproducible, synthetic bilingual training and held-out evaluation text.

Only original examples are used; no scraped data, copyrighted corpora,
secrets, or invented claims of pretrained knowledge. Questions about the same
math operands always belong to the same split across all phrasings/languages.
"""
import argparse
import hashlib
import random
from pathlib import Path

SEED = 20261017
FACTS = [
    ("What is LumoLM?", "LumoLM is an experimental language model."),
    ("Can you talk?", "I can generate text, but my abilities are limited."),
    ("Do you know Minecraft?", "Minecraft is a sandbox game about building and exploring."),
    ("What is a block in Minecraft?", "A block is a basic unit used to build the Minecraft world."),
    ("What is redstone?", "Redstone is used for circuits and mechanisms in Minecraft."),
    ("What is a creeper?", "A creeper is a hostile mob in Minecraft."),
    ("What is the Ender Dragon?", "The Ender Dragon is a boss in Minecraft."),
    ("What is a crafting table?", "A crafting table is used to craft items in Minecraft."),
    ("What is survival mode?", "Survival mode involves gathering resources and staying alive."),
    ("What is creative mode?", "Creative mode lets players build freely with many blocks."),
    ("What is a GPU?", "A GPU is a processor built for highly parallel computations."),
    ("What is a CPU?", "A CPU runs general-purpose instructions."),
    ("What is RAM?", "RAM holds data that programs are actively using."),
    ("What is a computer?", "A computer processes data using instructions."),
    ("What is an SSD?", "An SSD stores data on flash memory."),
    ("What is an operating system?", "An operating system manages hardware and software."),
    ("What is Linux?", "Linux is a family of operating systems built around the Linux kernel."),
    ("What is Python?", "Python is a programming language."),
    ("What is PyTorch?", "PyTorch is a library for tensor computations and machine learning."),
    ("What is a Transformer?", "A Transformer is a neural network architecture using attention."),
    ("What is a language model?", "A language model estimates probabilities of token sequences."),
    ("What is training?", "Training adjusts model parameters to reduce a loss function."),
    ("What is inference?", "Inference uses a trained model to produce predictions."),
    ("What is overfitting?", "Overfitting happens when a model memorizes data and generalizes poorly."),
    ("What is a tokenizer?", "A tokenizer converts text into tokens for a model."),
    ("What is a byte?", "A byte usually contains eight bits."),
    ("What is binary?", "Binary is a base-two number system."),
    ("What is hexadecimal?", "Hexadecimal is a base-sixteen number system."),
    ("What is a repository?", "A repository stores and tracks project files."),
    ("What is Git?", "Git is a version control system."),
    ("What is an API?", "An API is an interface for software communication."),
    ("What is JSON?", "JSON is a text format for structured data."),
    ("What does HTTP stand for?", "HTTP stands for Hypertext Transfer Protocol."),
    ("What is a network?", "A network connects devices for communication."),
    ("What is encryption?", "Encryption transforms data so only authorized parties can read it."),
    ("What is a database?", "A database stores and organizes information."),
    ("What is the sky?", "The sky is the atmosphere and space seen above Earth."),
    ("What is water?", "Water is a chemical compound made of hydrogen and oxygen."),
    ("Is the Earth flat?", "No, Earth is approximately spherical."),
    ("What is the Moon?", "The Moon is Earth's natural satellite."),
    ("What is the Sun?", "The Sun is the star at the center of our solar system."),
    ("How do plants grow?", "Plants need water, nutrients, and energy from light."),
    ("What is a kilogram?", "A kilogram is a unit of mass."),
    ("How many minutes are in an hour?", "There are 60 minutes in an hour."),
    ("How many seconds are in a minute?", "There are 60 seconds in a minute."),
    ("How many days are in a week?", "There are seven days in a week."),
    ("How do you greet someone?", "You can say hello."),
    ("Say hello.", "Hello!"),
    ("Say goodbye.", "Goodbye!"),
    ("How are you?", "I am a simple experimental program. Thank you for asking."),
    ("Who made you?", "I am a software experiment created by a developer."),
    ("Do you know today's date?", "I do not have a live clock in this model."),
    ("Can you access the internet?", "No, I cannot access the internet from this model."),
    ("Do you always answer correctly?", "No, I can make mistakes and generate nonsense."),
    ("Can you solve every problem?", "No, my model capacity and training are limited."),
    ("Can you explain a bug?", "Describe the bug and I can try to suggest a cause."),
    ("Do you have feelings?", "I am a text-generation program, not a person."),
    ("What is two plus two?", "Four."),
    ("Do you understand everything?", "No. This small model has very limited understanding."),
]
FACTS += [
    ("What is a keyboard?", "A keyboard is an input device for typing text."),
    ("What is a mouse?", "A mouse is an input device used to move a pointer."),
    ("What is a monitor?", "A monitor displays visual output from a computer."),
    ("What is a laptop?", "A laptop is a portable personal computer."),
    ("What is a motherboard?", "A motherboard connects many computer components."),
    ("What is a power supply?", "A power supply converts electrical power for computer parts."),
    ("What is a file?", "A file stores a sequence of data."),
    ("What is a folder?", "A folder groups files and other folders."),
    ("What is a process?", "A process is a running program."),
    ("What is a thread?", "A thread is a sequence of execution in a program."),
    ("What is multitasking?", "Multitasking allows tasks to progress while sharing resources."),
    ("What is a command line?", "A command line lets users enter text commands."),
    ("What is Bash?", "Bash is a command shell used on many Unix-like systems."),
    ("What is a function?", "A function is a reusable block of code."),
    ("What is a variable?", "A variable names a value in a program."),
    ("What is a loop?", "A loop repeats operations according to a condition."),
    ("What is a conditional?", "A conditional selects behavior based on a test."),
    ("What is a list?", "A list is an ordered collection of items."),
    ("What is a dictionary in Python?", "A dictionary maps keys to values."),
    ("What is a class?", "A class can define data and behaviors for objects."),
    ("What is an exception?", "An exception signals an error or unusual condition."),
    ("What is a unit test?", "A unit test checks a small piece of software."),
    ("What is version control?", "Version control tracks changes to files."),
    ("What is a commit?", "A commit records a snapshot of changes in Git."),
    ("What is a branch in Git?", "A branch is a movable name for a line of commits."),
    ("What is a pull request?", "A pull request proposes changes for review."),
    ("What is CI?", "CI automatically checks changes when code is updated."),
    ("What is open source?", "Open source software makes its source code available under a license."),
    ("What is a license?", "A software license defines permissions and restrictions."),
    ("What is a model checkpoint?", "A checkpoint stores model parameters and training state."),
    ("What is a gradient?", "A gradient describes how a quantity changes with input."),
    ("What is backpropagation?", "Backpropagation computes gradients through a neural network."),
    ("What is a learning rate?", "A learning rate controls the size of parameter updates."),
    ("What is a loss function?", "A loss function measures prediction error."),
    ("What is a batch?", "A batch is a group of training examples."),
    ("What is an epoch?", "An epoch is one pass over a training dataset."),
    ("What is validation data?", "Validation data measures performance on examples outside training."),
    ("What is a held-out test?", "A held-out test uses examples excluded from training."),
    ("What is a parameter?", "A parameter is a value learned or configured by a model."),
    ("What is an embedding?", "An embedding represents a token as a numeric vector."),
    ("What is attention?", "Attention combines information from different sequence positions."),
    ("What is causal attention?", "Causal attention prevents tokens from attending to future tokens."),
    ("What is a context window?", "A context window limits how many tokens a model sees at once."),
    ("What is token prediction?", "Token prediction estimates the next token in a sequence."),
    ("What is a prompt?", "A prompt is input text given to a language model."),
    ("What is temperature in sampling?", "Temperature controls randomness in token sampling."),
    ("What is a hallucination?", "A hallucination is generated content that is not well grounded."),
    ("What is an algorithm?", "An algorithm is a sequence of steps for solving a task."),
    ("What is a compiler?", "A compiler translates program code into another form."),
    ("What is an interpreter?", "An interpreter executes program instructions."),
    ("What is UTF-8?", "UTF-8 is a variable-width encoding of Unicode text."),
    ("What is ASCII?", "ASCII encodes a basic set of characters as numbers."),
    ("What is a pixel?", "A pixel is a small element of a digital image."),
    ("What is a frame rate?", "Frame rate is the number of images shown each second."),
    ("What is latency?", "Latency is the delay between an action and a response."),
    ("What is a server?", "A server supplies services to clients over a network."),
    ("What is a client?", "A client requests services from a server."),
    ("What is a port?", "A port helps distinguish network services on a device."),
    ("What is an IP address?", "An IP address identifies a network interface."),
    ("What is DNS?", "DNS maps human-readable domain names to network records."),
    ("What is HTTPS?", "HTTPS uses encryption to protect HTTP communication."),
    ("What is a firewall?", "A firewall filters network traffic according to rules."),
    ("What is a virtual machine?", "A virtual machine emulates a computer environment."),
    ("What is virtualization?", "Virtualization lets computing resources be divided into isolated environments."),
    ("What is a container?", "A container packages software with its runtime environment."),
    ("What is Vulkan?", "Vulkan is a low-level graphics and compute API."),
    ("What is OpenGL?", "OpenGL is a graphics programming API."),
    ("What is a shader?", "A shader is a program executed during graphics processing."),
    ("What is a Minecraft server?", "A Minecraft server hosts a shared game world."),
    ("What is a chunk in Minecraft?", "A chunk is a section of the Minecraft world."),
    ("What is a biome in Minecraft?", "A biome is a region with characteristic terrain and features."),
    ("What is a Nether portal?", "A Nether portal connects the Overworld and the Nether."),
    ("What is mining in Minecraft?", "Mining is collecting blocks and resources."),
    ("What is farming in Minecraft?", "Farming produces food or resources over time."),
    ("What is a furnace in Minecraft?", "A furnace smelts materials and cooks some foods."),
    ("What is an enchantment?", "An enchantment gives special effects to eligible items."),
    ("What is a villager in Minecraft?", "A villager is a non-player character that can trade."),
    ("What is an iron golem?", "An iron golem is a mob that can defend villagers."),
    ("What is a command block?", "A command block can execute Minecraft commands."),
    ("What is PvP?", "PvP means player versus player."),
    ("What does FPS mean?", "FPS means frames per second in graphics."),
]

CHINESE = [
    ("你是誰？", "我是 LumoLM，一個實驗語言模型。"),
    ("你好嗎？", "你好！我是 LumoLM。"),
    ("請打招呼。", "你好！"),
    ("你會中文嗎？", "我可以產生一些中文，但能力有限。"),
    ("你知道 Minecraft 嗎？", "Minecraft 是一款沙盒遊戲。"),
    ("什麼是紅石？", "Minecraft 中的紅石能用於製作電路與機關。"),
    ("什麼是電腦？", "電腦是根據指令處理資料的裝置。"),
    ("什麼是顯示卡？", "顯示卡負責處理圖形與平行計算工作。"),
    ("什麼是記憶體？", "記憶體暫時保存程式正在使用的資料。"),
    ("什麼是 Python？", "Python 是一種程式語言。"),
    ("什麼是作業系統？", "作業系統管理硬體與軟體資源。"),
    ("什麼是訓練？", "訓練會調整模型參數，以降低預測誤差。"),
    ("什麼是推理？", "推理是使用訓練好的模型產生預測。"),
    ("你會犯錯嗎？", "會。我可能產生錯誤或不合理的答案。"),
    ("你知道現在日期嗎？", "我的模型沒有即時時鐘。"),
    ("你可以上網嗎？", "不行。這個模型沒有網路工具。"),
    ("什麼是語言模型？", "語言模型預測文字序列中的下一個單位。"),
    ("什麼是資料庫？", "資料庫用於儲存與整理資訊。"),
    ("什麼是網路？", "網路讓不同裝置能夠交換資訊。"),
    ("什麼是位元組？", "一個位元組通常包含八個位元。"),
]
CHINESE += [
    ("什麼是滑鼠？", "滑鼠是一種用來控制指標的輸入裝置。"),
    ("什麼是鍵盤？", "鍵盤是一種用來輸入文字的裝置。"),
    ("什麼是主機板？", "主機板連接電腦中的許多零件。"),
    ("什麼是 CPU？", "CPU 是執行一般指令的處理器。"),
    ("什麼是 GPU？", "GPU 適合大量平行計算與圖形工作。"),
    ("什麼是 SSD？", "SSD 使用快閃記憶體保存資料。"),
    ("什麼是檔案？", "檔案用於保存資料。"),
    ("什麼是資料夾？", "資料夾用於整理檔案。"),
    ("什麼是伺服器？", "伺服器透過網路提供服務。"),
    ("什麼是 Git？", "Git 是版本控制工具。"),
    ("什麼是 GitHub？", "GitHub 是用來託管 Git 專案的平台。"),
    ("什麼是模型參數？", "模型參數是在訓練過程中調整的數值。"),
    ("什麼是上下文視窗？", "上下文視窗限制模型可同時處理的內容長度。"),
    ("什麼是過度擬合？", "過度擬合是記住訓練資料卻難以回答新問題。"),
    ("什麼是驗證資料？", "驗證資料用於測試模型對未知範例的表現。"),
    ("什麼是語彙單位？", "語彙單位是語言模型處理文字的基本片段。"),
    ("什麼是 Minecraft 生存模式？", "生存模式需要收集資源並維持生存。"),
    ("什麼是 Minecraft 創造模式？", "創造模式方便自由建造與探索。"),
    ("什麼是區塊？", "區塊是 Minecraft 世界的一部分。"),
    ("什麼是 PvP？", "PvP 是玩家對抗玩家。"),
]

EN_TEMPLATES = {
    "add": ("What is {a}+{b}?", "Calculate {a} + {b}.", "Add {a} and {b}.", "What does {a} plus {b} equal?"),
    "sub": ("What is {a}-{b}?", "Calculate {a} - {b}.", "Subtract {b} from {a}."),
    "mul": ("What is {a}*{b}?", "Multiply {a} by {b}.", "Calculate {a} times {b}."),
}
ZH_TEMPLATES = {
    "add": ("請計算 {a} 加 {b}。", "{a} 加 {b} 等於多少？"),
    "sub": ("請計算 {a} 減 {b}。", "{a} 減 {b} 等於多少？"),
    "mul": ("請計算 {a} 乘 {b}。", "{a} 乘 {b} 等於多少？"),
}


def held_out(kind: str, a: int, b: int) -> bool:
    # Ordered subtraction, canonical unordered addition and multiplication.
    if kind in ("add", "mul"):
        a, b = sorted((a, b))
    return (a * 43 + b * 19 + {"add": 11, "sub": 23, "mul": 31}[kind]) % 11 == 0


def make_pair(q: str, a: str) -> str:
    return f"Question: {q}\nAnswer: {a}\n"


def build():
    train, validation = [], []
    rng = random.Random(SEED)
    # All fixed facts are training-only; the validation set contains novel
    # arithmetic operands and special unseen prompts in the probe script.
    for q, a in FACTS + CHINESE:
        # Oversample clean explanations relative to short arithmetic lines, so
        # dialogue ability is not drowned out by easy-to-predict numbers.
        train.extend([make_pair(q, a)] * 18)
    for kind in ("add", "sub", "mul"):
        max_num = 40 if kind != "mul" else 12
        for a in range(max_num + 1):
            for b in range(max_num + 1):
                value = {"add": a + b, "sub": a - b, "mul": a * b}[kind]
                texts = [template.format(a=a, b=b) for template in EN_TEMPLATES[kind]]
                # Smaller Chinese data subset to avoid overwhelming byte model.
                if a <= 20 and b <= 20:
                    texts += [template.format(a=a, b=b) for template in ZH_TEMPLATES[kind]]
                target = validation if held_out(kind, a, b) else train
                rng.shuffle(texts)
                # Validation examples are unique pairs, never a changed phrasing
                # of a math pair already in the training corpus.
                selected = texts[:2] if target is validation else texts[::2]
                for q in selected:
                    target.append(make_pair(q, str(value)))
    rng.shuffle(train)
    rng.shuffle(validation)
    return "".join(train).encode("utf-8"), "".join(validation).encode("utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, default=Path("data/expanded_train.txt"))
    p.add_argument("--val", type=Path, default=Path("data/expanded_val.txt"))
    args = p.parse_args()
    train, val = build()
    if not train or not val:
        raise RuntimeError("dataset split must not be empty")
    args.train.parent.mkdir(parents=True, exist_ok=True)
    args.val.parent.mkdir(parents=True, exist_ok=True)
    args.train.write_bytes(train)
    args.val.write_bytes(val)
    print(f"train_bytes={len(train)} train_sha256={hashlib.sha256(train).hexdigest()}")
    print(f"val_bytes={len(val)} val_sha256={hashlib.sha256(val).hexdigest()}")
    print("Dataset source: original synthetic dialogue examples, held-out math operand pairs.")


if __name__ == "__main__":
    main()
