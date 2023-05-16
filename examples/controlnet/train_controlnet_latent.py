#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.17.0.dev0")

logger = get_logger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def log_validation(vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    controlnet = accelerator.unwrap_model(controlnet)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []

    if args.validation_condition_latent is not None:
        validation_condition_latent = torch.tensor([0.8126324415206909, -0.7309863567352295, 0.225254088640213, 0.09128245711326599, 0.7010924816131592, 0.8665623664855957, -0.6165128350257874, 0.27650102972984314, -0.0005173508543521166, 0.8779447674751282, -0.01689322665333748, -0.37897539138793945, -0.36857613921165466, 0.18431393802165985, -0.25083327293395996, 0.711447536945343, 0.30887383222579956, 0.8545997738838196, 0.5815681219100952, 0.12503348290920258, 0.7152705788612366, -0.2090848833322525, -0.8724581003189087, -0.25837117433547974, 0.6435180902481079, 1.0367200374603271, 0.31880950927734375, -0.08400630950927734, 0.3737243711948395, 0.10326693952083588, -0.07969196140766144, -0.1306680291891098, -0.1077885702252388, -0.008828987367451191, -0.06404285877943039, 0.18567903339862823, -0.006228245794773102, -0.546561598777771, 0.30891168117523193, -0.821702778339386, 1.065394639968872, 0.19913093745708466, 0.4547443389892578, -0.6596407890319824, 0.26950758695602417, -0.2973967492580414, -0.04768631234765053, -0.14702895283699036, 0.10624386370182037, 0.7451660633087158, -0.1501137614250183, -0.11482361704111099, 0.05834248661994934, 0.3541406989097595, 0.18997782468795776, 0.8142974972724915, -0.8772384524345398, 1.3620091676712036, -0.8844830989837646, 0.4732404053211212, 0.4692832827568054, -0.2539037764072418, 0.08240831643342972, -0.5067110657691956, 0.2087959498167038, -0.4408934712409973, 0.5680832862854004, 0.17710992693901062, 0.6582611203193665, 0.19978247582912445, -0.10140694677829742, 0.5152105689048767, 0.14230592548847198, 0.31664544343948364, -2.163442373275757, -0.5536922812461853, -0.37738412618637085, 0.33281195163726807, 0.11710302531719208, -1.1364997625350952, 0.5778387188911438, -0.734591007232666, -1.1479432582855225, 0.7521857619285583, 0.7643232941627502, 0.16426531970500946, 0.9403416514396667, 0.03545738384127617, 0.10644323378801346, 0.2226894497871399, 0.7498921751976013, -0.10992506146430969, -0.10616622120141983, 0.35408076643943787, 0.5485960245132446, -0.07544509321451187, 1.120057225227356, 0.30357107520103455, -0.470552921295166, 0.6569657921791077, 0.8924034833908081, 0.7777339220046997, -1.1219701766967773, -0.5306450724601746, -0.7035248875617981, -0.09352106600999832, -0.09973932802677155, -0.6663515567779541, -0.35797739028930664, -0.32294541597366333, 1.0605295896530151, 1.6086643934249878, -0.3605612814426422, 0.3929201066493988, 0.11061322689056396, -0.21202488243579865, 0.5557827353477478, -0.1514606475830078, -0.6534454822540283, 0.9851502180099487, -0.2463056743144989, -0.5635208487510681, 0.15911537408828735, 1.0791356563568115, 0.37972143292427063, -0.05061066150665283, 1.0522745847702026, -0.13346894085407257, 1.1640894412994385, 0.2690299153327942, -0.2696649432182312, 0.8959578275680542, -0.5897513628005981, 0.3952793776988983, -0.1861565113067627, 1.3385343551635742, -0.34535637497901917, -0.5774950385093689, -0.12241347134113312, 0.2934127449989319, 0.17476089298725128, -0.2337692677974701, 0.3306209146976471, 0.042865999042987823, 0.1659477949142456, 0.3621094226837158, 0.9392814636230469, -0.20433735847473145, 0.6918020844459534, -0.7584225535392761, 0.19210369884967804, -0.4108317792415619, 0.7848499417304993, 0.11341312527656555, -0.7635840773582458, 0.5362327098846436, -0.4243270456790924, -0.34766900539398193, 0.546959638595581, 0.4718921184539795, 0.6139020323753357, 0.6683499217033386, -0.5414732694625854, -1.0489914417266846, 0.2694435119628906, 0.3982829749584198, -0.25143298506736755, -0.051351528614759445, 0.45057931542396545, -0.8712934851646423, -0.19956474006175995, -0.7502299547195435, 0.0988396555185318, 0.653100311756134, -0.22549034655094147, -1.184130311012268, 0.7721866369247437, -0.36379724740982056, 0.7874416708946228, -1.332216739654541, 1.3400013446807861, -0.7588857412338257, -0.9744035601615906, 0.06277799606323242, -0.20532748103141785, -0.1955127716064453, 0.10064925253391266, 0.44887855648994446, 0.013208802789449692, -1.0174862146377563, 0.3168657720088959, 0.09333425760269165, -0.6070816516876221, 0.6156541705131531, 0.17384719848632812, 0.6075625419616699, -0.10104870796203613, 0.8008090853691101, 0.5896300673484802, -0.694779098033905, -0.18317638337612152, 0.28983059525489807, 0.27486053109169006, -0.31687191128730774, -0.12073266506195068, -0.5329442620277405, -1.0577553510665894, 0.11422920972108841, -0.2534070312976837, -1.3130409717559814, 0.06682252883911133, -0.15802106261253357, -0.3052307665348053, -0.9422062039375305, -0.2865491807460785, 1.062909722328186, 0.906088650226593, -0.637313723564148, -0.6529932022094727, -0.15596513450145721, -0.3021138310432434, -0.2998255491256714, 0.8978193402290344, 0.006722019985318184, 0.956364095211029, 1.6968963146209717, -0.6611747741699219, 0.17399317026138306, 0.21101625263690948, 0.08144601434469223, 0.4340282082557678, 0.4623401165008545, 0.44270455837249756, 0.038042303174734116, 0.6905386447906494, 0.0857178196310997, -0.02174047753214836, -0.0158642940223217, -0.4546254873275757, -0.3986422121524811, 0.5545988082885742, 0.39961186051368713, -0.013855554163455963, -0.43660709261894226, 0.023886486887931824, 0.34229132533073425, 0.1892988383769989, -0.6332226991653442, 0.0829254761338234, 0.005790417082607746, 1.6472147703170776, -0.26822876930236816, 0.5203754305839539, -0.31707462668418884, 0.06190422177314758, 0.8669780492782593, -0.29112160205841064, 0.09207939356565475, 0.1827765554189682, -0.17766177654266357, -1.1350581645965576, 1.500321626663208, 0.5137163996696472, 0.4199111759662628, 0.45953819155693054, 0.513347864151001, -0.765156626701355, -0.7077879905700684, -0.5513373017311096, -0.08539849519729614, 1.1885966062545776, -0.5503196716308594, -0.12287856638431549, -0.07175296545028687, 0.28016382455825806, 0.4748111069202423, 0.530887246131897, 0.06284146010875702, 0.33531850576400757, -0.060291483998298645, -0.3632679581642151, 0.2364622801542282, -0.9446687698364258, 0.18639513850212097, 0.44261664152145386, -0.29192644357681274, 0.6980015635490417, 0.4558507204055786, -0.6164469718933105, -0.3952762186527252, 0.6473113894462585, -0.35740843415260315, 0.14331264793872833, 0.44560348987579346, 0.2767159044742584, -0.6929587721824646, 0.09245389699935913, 0.33786138892173767, -0.546709418296814, -0.1463910937309265, 0.17548783123493195, 0.21632574498653412, -0.7905657291412354, -0.2249501645565033, 1.0625193119049072, 0.895283043384552, -0.5955145955085754, -0.12196753174066544, 0.1605595052242279, 0.5071552991867065, -0.5248556137084961, 0.37145155668258667, 0.5401180982589722, -0.077814020216465, 0.7601260542869568, 1.530561923980713, 0.5888325572013855, 0.12136220186948776, -0.8615868091583252, -1.1322112083435059, 1.026488184928894, 0.08864481747150421, -0.7178512811660767, 0.6560743451118469, -0.8985272645950317, -0.17436742782592773, 0.30583396553993225, -0.017820384353399277, 0.14365054666996002, -0.16381143033504486, 0.2563571333885193, 0.3886145353317261, 0.2629714906215668, -0.08838865160942078, 0.7708043456077576, -0.49397408962249756, 0.2826741933822632, -0.6731964349746704, -0.8037588000297546, 0.20337827503681183, -1.9102485179901123, -0.8921554684638977, 0.20966371893882751, -0.011821700260043144, 0.18817570805549622, -0.013673914596438408, -0.83927983045578, 0.3619507849216461, 0.617149829864502, 0.11515526473522186, 0.37758728861808777, -0.3319835662841797, -0.7013740539550781, -0.02145284041762352, 0.17362180352210999, 0.1296091377735138, 0.12693090736865997, -0.17997322976589203, 0.28261908888816833, -0.7659951448440552, -0.15637367963790894, -0.408830851316452, -1.0638965368270874, 0.7950081825256348, -0.3435124158859253, -0.4348413646221161, 0.16184040904045105, -0.6790182590484619, -0.11140680313110352, -0.5899059176445007, -0.010403972119092941, -1.016640067100525, 0.34052202105522156, 1.227500319480896, 0.06828194856643677, 0.68144291639328, -0.6343870162963867, -0.23536905646324158, -0.30176180601119995, -0.2833588123321533, 0.18958377838134766, -0.33892542123794556, -1.1151498556137085, 0.6592872142791748, -0.05830524489283562, -0.2849886417388916, -0.46094971895217896, 0.20390768349170685, -0.4301096200942993, 0.407546728849411, 0.6487147808074951, -0.07003768533468246, 0.08241862058639526, 0.5011270046234131, 0.9270802736282349, 0.2669534683227539, 0.16854703426361084, 0.280252069234848, -0.37060898542404175, 0.15955664217472076, 0.40732964873313904, 0.4899470806121826, 0.6773754358291626, -0.9965454339981079, -1.066056728363037, -0.05641596391797066, -0.6278991103172302, 0.8716244101524353, -0.0010716044344007969, 0.896540641784668, -0.11364331096410751, -0.5657854080200195, -0.8619548678398132, -0.1329343616962433, -0.9237490296363831, -0.4596918523311615, 0.04255709797143936, -0.19260059297084808, 0.3823148012161255, -0.12242703139781952, 0.13086792826652527, -0.7797015309333801, 0.6495129466056824, 0.06012086197733879, -0.5161685943603516, -0.06111528351902962, -0.28437739610671997, -0.5889847874641418, -0.29426658153533936, 0.14381825923919678, -0.7615663409233093, 0.8820446729660034, -0.2194739282131195, -0.0006138760363683105, 0.6913018226623535, 0.43502378463745117, 1.1496531963348389, 0.2957896888256073, 0.6691354513168335, -0.07920592278242111, -0.20974665880203247, -0.7237433791160583, 0.7316288352012634, -0.2517208456993103, 1.5054948329925537, -0.09355089068412781, -0.2396278977394104, 0.18065330386161804, -0.4590109884738922, 0.6450765132904053, -1.1068177223205566, 0.4962785542011261, 0.31050899624824524, 0.4759293496608734, 1.0582329034805298, -0.3042311668395996, -0.2839759886264801, -0.7300487756729126, 0.7861716747283936, -0.07047657668590546, 1.5670157670974731, -1.1404098272323608, -0.21763773262500763, -0.15642297267913818, -0.3245178759098053, -0.7258021831512451, 0.0001641832059249282, 0.147551491856575, 0.11251600831747055, -0.29038122296333313, 0.509395182132721, -0.753182053565979, -0.7679516077041626, -0.22310657799243927, 0.4672211706638336, -0.4169514775276184, -0.4365799129009247, -0.0386040173470974, -0.1427774429321289, 0.46317777037620544, 0.4356531500816345, -0.6227076649665833, 0.3932393491268158, 0.12725615501403809, 0.026585105806589127, 0.9184682369232178, 1.4808731079101562, 0.09268134832382202, 0.684745728969574, -0.12288161367177963, -0.9230278730392456, -0.8679928183555603, -0.37566864490509033, -0.7360190153121948, 0.47442078590393066, 0.47829803824424744, -0.6699380874633789, -2.176067590713501, -0.23211811482906342, -0.42285385727882385, 1.1539437770843506, 0.6339414715766907, -0.8579012155532837, 0.6810212731361389, 0.4678162932395935, -0.46021905541419983, -0.045632652938365936, 0.8786377310752869, -0.2115039974451065, 0.767040491104126, -0.6232236623764038, 0.10577905178070068, 0.7326064705848694, -0.3698746860027313, 0.29791855812072754, 1.0444992780685425, -0.9928895235061646, 0.08095697313547134, -0.10530367493629456, -0.6028133630752563, 0.545669436454773, -0.0848260372877121, -0.2126971185207367, 0.06993890553712845, -0.4241849184036255, -0.06015315651893616, 0.5129142999649048, 0.0899580866098404, -0.8182541728019714, 0.4377489387989044, 0.18239963054656982, 0.19915838539600372, 0.7719345688819885, 0.32759931683540344, 0.7197173237800598, 0.5833828449249268, 1.3459187746047974, -0.6843072175979614, -0.46488088369369507, 0.11050569266080856, -0.09713239967823029, 1.2284432649612427, 0.05134344846010208, 0.19603903591632843, -1.34125816822052, 0.2294415980577469, 0.7930886149406433, 0.1439526528120041, 0.8017337918281555, 0.4039044976234436, -0.3509572744369507, -0.19603696465492249, -1.311174988746643, -0.42914530634880066, 0.2767788767814636, 0.7400491237640381, 0.7016057372093201, 0.14649739861488342, -0.1419687718153, 0.16781844198703766, -0.3530834913253784, 0.01751641370356083, 0.4376915991306305, -0.45269495248794556, 0.019198792055249214, -0.7755883932113647, 0.21652592718601227, -0.41550901532173157, 0.6593974232673645, 1.3374627828598022, 0.002920521656051278, 0.5732763409614563, 0.43351659178733826, 0.3469920754432678, 0.35130569338798523, -1.7005246877670288, 0.43032440543174744, -0.039348799735307693, 0.833774745464325, -0.12251376360654831, -0.7611007690429688, -0.032529860734939575, -0.29708802700042725, -0.08286096155643463, 0.37914711236953735, -0.8773914575576782, 0.5748604536056519, -0.8973140120506287, -0.3752395808696747, -0.7538363933563232, -0.04385518282651901, 0.0811978429555893, -0.09775058925151825, 0.08406960964202881, -0.25785112380981445, -0.37120604515075684, -1.0603842735290527, -0.10671243816614151, -0.42561572790145874, -0.15385876595973969, 0.17607751488685608, -0.23935559391975403, -0.05955400690436363, 1.1473244428634644, 0.8445087671279907, -0.1411619484424591, 0.33923035860061646, -0.08367978036403656, 0.16346915066242218, 0.657371997833252, -0.3325171172618866, -0.9234058856964111, -0.5299535989761353, 0.12482312321662903, 0.5029850006103516, 0.5199931859970093, -0.10944624990224838, -0.790457546710968, -0.21285536885261536, -0.47584155201911926, -0.7981476783752441, 1.0695035457611084, 0.2610572278499603, 0.5881476402282715, -0.8678709864616394, -0.35038766264915466, -1.2579296827316284, -0.6274574398994446, -0.6413892507553101, -0.28417104482650757, 0.7796409130096436, 0.8546446561813354, 1.2243220806121826, -0.7298498153686523, 0.23374797403812408, 0.8350761532783508, 0.6228645443916321, 0.28415149450302124, 0.19806720316410065, 0.1583377867937088, -0.7402756810188293, -0.3335207402706146, 0.24422287940979004, -0.4623316824436188, -0.6122536063194275, 0.556064784526825, 0.6569390296936035, -0.8050475716590881, -0.937269926071167, 0.2851131558418274, 1.1544992923736572, -1.0594463348388672, 0.5736433863639832, 0.2216963768005371, -0.19582445919513702, 1.1432316303253174, -0.41876232624053955, 1.0772234201431274, 0.21115607023239136, -1.0963187217712402, -0.2822684049606323, 0.16009509563446045, -0.44885963201522827, 0.2630183696746826, -0.8575333952903748, 1.0636509656906128, 0.36558136343955994, 0.035985082387924194, 0.3809289336204529, 0.16477371752262115, 0.436871737241745, 0.5455122590065002, 0.06469198316335678, 0.39084652066230774, 0.4920084774494171, 0.3564370572566986, 0.5475119352340698, -1.2495555877685547, 0.5858922600746155, 0.9195133447647095, 0.5087279677391052, 0.48088064789772034, 0.013999875634908676, 0.38659289479255676, -0.5578427314758301, -0.2840521037578583, -0.5762179493904114, -0.11450442671775818, 0.023169053718447685, -0.060240887105464935, 0.23728151619434357, 0.4556715786457062, -0.061392124742269516, 0.37635841965675354, -0.16077855229377747, -0.46703240275382996, 0.5645114779472351, 0.08885245770215988, -1.3540538549423218, -0.4145331382751465, -0.6197600364685059, -0.35897934436798096, -0.5693356990814209, -0.9159308671951294, 0.17020384967327118, -0.8742132782936096, -0.6508145332336426, 0.555121898651123, -0.8664371967315674, -0.18187706172466278, 0.24996593594551086, 0.41669508814811707, 1.0120142698287964, 0.19638797640800476, -0.3061444163322449, -0.3499210476875305, 0.035331111401319504, -1.3083282709121704, -0.7315471768379211, 1.0008143186569214, -0.29280373454093933, -1.3123668432235718, 0.8331720232963562, -0.22620031237602234, 0.09221073985099792, -0.3001444637775421, -0.1590837836265564, 0.6579075455665588, -0.9231489896774292, 0.5603455305099487, 0.28364452719688416, -0.014060750603675842, -0.959665060043335, 0.0362476147711277, -0.7197028398513794, 0.4179375469684601, 0.571057140827179, 0.7430841326713562, 1.1780989170074463, 0.28666573762893677, 0.3802201747894287, 1.4691916704177856, 0.014460468664765358, -1.2291511297225952, -0.5971983075141907, -0.8741455674171448, -0.1150544285774231, 0.3249701261520386, 0.5373644232749939, -0.23526552319526672, -0.6829221248626709, 0.8629854917526245, 0.12341640889644623, 1.5390095710754395, -1.4176539182662964, 0.2964009642601013, 1.089040994644165, 0.31779763102531433, 0.572288990020752, -0.8800629377365112, -0.7783003449440002, 0.3810289204120636, -0.1867397129535675, 0.42904698848724365, 0.08234770596027374, -0.1227588951587677, -0.36463531851768494, -0.09393692016601562])
        validation_condition_latent = validation_condition_latent.repeat(77,1) 
        validation_condition_latent = validation_condition_latent.unsqueeze(0).cuda()


    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        validation_image = Image.open(validation_image).convert("RGB")

        images = []

        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                if args.validation_condition_latent is not None:
                    image = pipeline(
                        validation_prompt,
                        validation_image,
                        num_inference_steps=20,
                        generator=generator,
                        conditioning_latent=validation_condition_latent
                    ).images[0]
                else:
                    image = pipeline(
                        validation_prompt, validation_image, num_inference_steps=20, generator=generator
                    ).images[0]

            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        return image_logs


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--validation_condition_latent",
        type=list,
        default=None,
        help="A list of latent vectors to be used for validation. If not specified, no latents will be used.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def make_train_dataset(args, tokenizer, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir,
                cache_dir=args.cache_dir,
            )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[3]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[1]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
        
    conditioning_latent_column = column_names[2]
    logger.info(f"conditioning latent column defaulting to {conditioning_latent_column}")


    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if random.random() < args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    #Expand the latent dimension from 768 to 77*768 to match the size of the encoder_hidden_states
    def expand_latent(latent):
        latent_tensor = torch.tensor(latent)
        return latent_tensor 

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        conditioning_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["input_ids"] = tokenize_captions(examples)
        examples["conditioning_latent"] =[expand_latent(latent) for latent in examples[conditioning_latent_column]]

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    conditioning_latent = torch.stack([example["conditioning_latent"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
        "conditioning_latent": conditioning_latent,
    }


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            i = len(weights) - 1

            while len(weights) > 0:
                weights.pop()
                model = models[i]

                sub_dir = "controlnet"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = make_train_dataset(args, tokenizer, accelerator)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                #Get the image latent space for conditioning
                controlnet_latents = batch["conditioning_latent"].to(dtype=weight_dtype)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=controlnet_latents,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(controlnet)
        controlnet.save_pretrained(args.output_dir)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
