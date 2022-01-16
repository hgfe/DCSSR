# DCSSR



## Da Vinci Dataset

https://drive.google.com/drive/folders/1ov7tX916DrPRqrsaaKInR-jLtQHgVPAe?usp=sharing


## 5-fold Cross Validation Results

### da Vinci Dataset (x2)

PSNR
|  Fold   | 1 | 2 | 3 | 4 | 5 | avg |
|:----------:|:----:|:----:|:----:|:----:|:----:|:----:|
|bicubic  |37.7731±0.5440|36.9459±0.8474|37.0630±0.5632|37.7490±1.2220|38.2665±1.1444|37.5595±0.8642|
|SRCNN    |41.6352±0.5763|40.9319±0.9536|40.9069±0.5482|41.7738±1.1885|41.6496±1.2996|41.3795±0.9132|
|VDSR     |42.2607±0.5331|41.7304±0.8328|41.7068±0.5147|42.4065±1.0917|42.0082±1.3634|42.0225±0.8671|
|DRRN     |42.0806±0.5521|41.7680±0.8843|41.4216±0.5245|42.2336±1.1579|42.1917±1.3461|41.9391±0.8930|
|StereoSR |42.0639±0.5477|41.7154±0.8739|41.4031±0.5344|42.3086±1.1105|42.2135±1.3671|41.9409±0.8867|
|PASSR    |42.3387±0.5337|41.9255±0.8311|41.7217±0.5447|42.1344±1.0566|42.0639±1.2710|42.0368±0.8474|
|DCSSR    |42.3822±0.5206|41.9016±0.8178|41.9057±0.5393|42.4578±1.0976|42.2875±1.3292|42.1870±0.8609|
|a1       |42.3172±0.5424|41.7715±0.8287|41.8643±0.5396|42.3508±1.0962|42.1697±1.3731|42.0947±0.8760|
|a2       |42.339±0.52490|41.8738±0.8215|41.9521±0.5400|42.3714±1.0981|42.2462±1.3595|42.1565±0.8688|
|a3       |42.2963±0.5258|41.8531±0.8065|41.7328±0.5470|42.3913±1.1074|42.2094±1.3685|42.0966±0.8710|
|a4       |42.3514±0.5546|41.8982±0.8884|41.7974±0.5387|42.3230±1.0860|42.2447±1.3167|42.1229±0.8769|
 
SSIM
|  Fold   | 1 | 2 | 3 | 4 | 5 | avg |
|:----------:|:----:|:----:|:----:|:----:|:----:|:----:|
|bicubic  |0.9913±0.0007|0.9910±0.0011|0.9911±0.0006|0.9913±0.0014|0.9911±0.0017|0.9912±0.0011|
|SRCNN    |0.9948±0.0004|0.9948±0.0005|0.9947±0.0004|0.9949±0.0008|0.9948±0.0008|0.9948±0.0006|
|VDSR     |0.9954±0.0004|0.9954±0.0005|0.9954±0.0003|0.9954±0.0008|0.9949±0.0010|0.9953±0.0006|
|DRRN     |0.9952±0.0004|0.9954±0.0005|0.9952±0.0003|0.9953±0.0008|0.9949±0.0010|0.9952±0.0006|
|StereoSR |0.9953±0.0004|0.9953±0.0004|0.9952±0.0003|0.9953±0.0008|0.9950±0.0010|0.9952±0.0006|
|PASSR    |0.9956±0.0004|0.9957±0.0004|0.9956±0.0003|0.9955±0.0007|0.9950±0.0009|0.9955±0.0005|
|DCSSR    |0.9956±0.0004|0.9958±0.0004|0.9958±0.0003|0.9957±0.0007|0.9952±0.0009|0.9956±0.0005|
|a1       |0.9955±0.0004|0.9956±0.0004|0.9957±0.0003|0.9957±0.0007|0.9950±0.0010|0.9955±0.0006|
|a2       |0.9955±0.0004|0.9958±0.0004|0.9957±0.0003|0.9957±0.0007|0.9951±0.0009|0.9956±0.0005|
|a3       |0.9955±0.0004|0.9957±0.0004|0.9957±0.0004|0.9957±0.0007|0.9951±0.0010|0.9955±0.0006|
|a4       |0.9955±0.0004|0.9956±0.0005|0.9955±0.0003|0.9956±0.0007|0.9951±0.0009|0.9955±0.0006|

### da Vinci Dataset (x4)

PSNR
|  Fold   | 1 | 2 | 3 | 4 | 5 | avg |
|:----------:|:----:|:----:|:----:|:----:|:----:|:----:|
|bicubic  |30.1586±0.5456|29.3778±0.7788|29.4477±0.4780|30.4423±1.3088|30.9610±1.2139|30.0775±0.8650|
|SRCNN    |32.3764±0.5749|31.7261±0.8606|31.6893±0.5221|32.7356±1.3787|32.9426±1.4032|32.2940±0.9479|
|VDSR     |33.7391±0.5958|33.1896±0.8543|32.9260±0.6042|33.8686±1.2081|33.2273±1.5192|33.3901±0.9563|
|DRRN     |33.4324±0.6202|32.9138±0.9178|32.9788±0.5764|33.8427±1.3644|33.4005±1.4776|33.3136±0.9913|
|StereoSR |33.3715±0.6035|33.0380±0.9056|32.6205±0.6237|33.5795±1.2177|33.3171±1.4903|33.1853±0.9682|
|PASSR    |34.0912±0.5776|33.6914±0.9202|33.3781±0.6590|34.0157±1.1654|33.3002±1.4847|33.6953±0.9614|
|DCSSR    |34.3103±0.5930|33.8040±0.8740|33.6344±0.6730|34.1342±1.1608|33.4410±1.5328|33.8648±0.9667|
|a1       |34.2138±0.6043|33.6971±0.8971|33.4072±0.6785|34.0565±1.1748|33.3657±1.5865|33.7481±0.9882|
|a2       |34.2504±0.5863|33.7392±0.8819|33.6295±0.6750|34.1096±1.1834|33.4178±1.5395|33.8293±0.9732|
|a3       |34.2091±0.6015|33.6278±0.9060|33.3951±0.6821|34.1079±1.1697|33.3680±1.5619|33.7416±0.9842|
|a4       |34.2709±0.5994|33.7416±0.8908|33.4735±0.6680|34.0852±1.1332|33.3934±1.5363|33.7929±0.9655|

SSIM
|  Fold   | 1 | 2 | 3 | 4 | 5 | avg |
|:----------:|:----:|:----:|:----:|:----:|:----:|:----:|
|bicubic  |0.9549±0.0048|0.9518±0.0060|0.9518±0.0028|0.9563±0.0079|0.9756±0.0090|0.9581±0.0061|
|SRCNN    |0.9691±0.0031|0.9686±0.0041|0.9684±0.0019|0.9706±0.0057|0.9701±0.0068|0.9694±0.0043|
|VDSR     |0.9760±0.0025|0.9760±0.0033|0.9744±0.0019|0.9759±0.0048|0.9730±0.0065|0.9751±0.0036|
|DRRN     |0.9753±0.0027|0.9756±0.0036|0.9749±0.0018|0.9760±0.0050|0.9731±0.0064|0.9750±0.0036|
|StereoSR |0.9742±0.0026|0.9748±0.0033|0.9735±0.0019|0.9749±0.0049|0.9727±0.0067|0.9740±0.0039|
|PASSR    |0.9769±0.0024|0.9776±0.0031|0.9768±0.0020|0.9770±0.0045|0.9729±0.0063|0.9762±0.0037|
|DCSSR    |0.9780±0.0024|0.9785±0.0029|0.9775±0.0020|0.9775±0.0045|0.9733±0.0066|0.9770±0.0037|
|a1       |0.9773±0.0024|0.9782±0.0029|0.9773±0.0020|0.9771±0.0045|0.9730±0.0065|0.9766±0.0037|
|a2       |0.9777±0.0024|0.9785±0.0028|0.9774±0.0020|0.9774±0.0045|0.9732±0.0065|0.9768±0.0036|
|a3       |0.9774±0.0024|0.9784±0.0029|0.9773±0.0020|0.9773±0.0044|0.9730±0.0066|0.9767±0.0037|
|a4       |0.9776±0.0025|0.9784±0.0028|0.9774±0.0020|0.9773±0.0044|0.9731±0.0063|0.9768±0.0036|


### Medtronic Dataset (x2)

PSNR
|  Fold   | 1 | 2 | 3 | 4 | 5 | avg |
|:----------:|:----:|:----:|:----:|:----:|:----:|:----:|
|bicubic  |42.6840±1.0572|42.4703±1.0074|42.2554±0.8949|43.2235±1.1999|42.3013±1.2207|42.5869±1.0760|
|SRCNN    |45.7139±0.9981|45.0568±1.0630|45.0832±0.8152|46.1065±1.0828|45.9805±1.4075|45.5882±1.0733|
|VDSR     |47.3065±0.927 |46.1190±1.3706|45.9982±0.9395|47.1095±1.0492|46.8157±1.6146|46.6698±1.1802|
|DRRN     |47.3890±0.8971|46.0739±1.2109|45.9736±0.9132|47.1229±1.0365|46.9233±1.3508|46.6965±1.0817|
|StereoSR |45.8067±0.9335|45.1639±1.1392|45.9821±0.9365|46.3782±1.0332|46.1039±1.4508|45.8870±1.0986|
|PASSR    |45.7781±1.0111|45.0484±1.1630|46.0572±0.9720|46.4635±0.8524|46.5194±1.4919|45.9733±1.0981|
|DCSSR    |47.5027±0.9450|46.3988±0.9325|46.1625±0.8798|47.4845±1.0562|47.0817±1.5371|46.9260±1.0701|
|a1       |47.3021±0.9706|46.1436±0.9445|45.8920±0.8921|47.1013±1.0933|46.8239±1.5186|46.6526±1.0838|
|a2       |47.4873±0.9820|46.3109±0.9413|46.0557±0.9026|47.4082±0.9983|46.9845±1.5340|46.8493±1.0716|
|a3       |47.3209±0.9603|46.2842±0.9336|46.0182±0.8895|47.3591±1.0602|46.8336±1.5477|46.7632±1.0783|
|a4       |47.3709±0.9776|46.2931±0.9395|46.1025±0.9032|47.3628±0.9981|46.8774±1.5360|46.8013±1.0709|


SSIM
|  Fold   | 1 | 2 | 3 | 4 | 5 | avg |
|:----------:|:----:|:----:|:----:|:----:|:----:|:----:|
|bicubic  |0.9946±0.0005|0.9942±0.0008|0.9945±0.0007|0.9950±0.0006|0.9945±0.0008|0.9946±0.0007|
|SRCNN    |0.9950±0.0011|0.9946±0.0009|0.9958±0.0007|0.9957±0.0006|0.9961±0.0013|0.9954±0.0009|
|VDSR     |0.9967±0.0006|0.9961±0.0007|0.9964±0.0007|0.9968±0.0005|0.9963±0.0010|0.9965±0.0007|
|DRRN     |0.9966±0.0005|0.9960±0.0007|0.9963±0.0007|0.9968±0.0005|0.9965±0.0010|0.9964±0.0007|
|StereoSR |0.9952±0.0005|0.9947±0.0009|0.9961±0.0006|0.9961±0.0006|0.9962±0.0008|0.9957±0.0007|
|PASSR    |0.9950±0.0008|0.9946±0.0018|0.9963±0.0007|0.9962±0.0005|0.9963±0.0007|0.9957±0.0009|
|DCSSR    |0.9969±0.0005|0.9965±0.0007|0.9967±0.0006|0.9971±0.0005|0.9968±0.0006|0.9968±0.0006|
|a1       |0.9966±0.0005|0.9960±0.0007|0.9964±0.0006|0.9968±0.0005|0.9965±0.0007|0.9965±0.0006|
|a2       |0.9969±0.0005|0.9964±0.0008|0.9966±0.0006|0.9971±0.0005|0.9967±0.0007|0.9967±0.0006|
|a3       |0.9968±0.0005|0.9964±0.0007|0.9964±0.0006|0.9970±0.0005|0.9966±0.0006|0.9966±0.0006|
|a4       |0.9968±0.0005|0.9964±0.0008|0.9965±0.0006|0.9970±0.0005|0.9966±0.0007|0.9967±0.0006|

### Medtronic Dataset (x4)

PSNR
|  Fold   | 1 | 2 | 3 | 4 | 5 | avg |
|:----------:|:----:|:----:|:----:|:----:|:----:|:----:|
|bicubic  |37.0274±0.9695|36.5333±0.7723|36.6439±0.8096|37.1620±1.0968|36.7522±1.2742|36.8238±0.9845|
|SRCNN    |40.3018±0.9611|39.9554±0.8164|40.1760±0.9504|40.3817±1.1977|40.0168±1.0230|40.1663±0.9897|
|VDSR     |40.7982±0.9658|40.7563±0.9348|41.0932±1.1372|40.9859±1.2356|40.5966±1.1855|40.8460±1.0918|
|DRRN     |41.1697±0.9216|40.8581±0.8671|41.0318±1.0485|41.0925±1.3320|40.8674±1.1506|41.0039±1.0640|
|StereoSR |40.4599±0.9455|40.0173±0.8692|40.7911±1.1167|40.7562±1.2054|40.2780±1.1429|40.4605±1.0559|
|PASSR    |40.8536±0.9430|40.6269±0.9336|40.8472±1.0588|40.9334±1.2932|40.6593±1.1749|40.7841±1.0807|
|DCSSR    |41.3208±0.9773|40.9832±0.8194|41.0377±1.1603|41.3087±1.3194|41.0621±1.0832|41.1425±1.0719|
|a1       |41.0299±0.9501|40.7763±0.8874|40.9051±1.0932|41.1069±1.2908|40.8237±1.1097|40.9284±1.0662|
|a2       |41.1517±0.9205|40.8648±0.8941|41.0431±1.1493|41.2930±1.2855|41.0155±1.1136|41.0736±1.0726|
|a3       |41.0937±0.9416|40.8021±0.8973|41.0252±1.1031|41.0943±1.2880|40.8340±1.0941|40.9699±1.0648|
|a4       |41.0866±0.9623|40.8594±0.9021|41.0130±1.1149|41.1056±1.2766|40.9824±1.0859|41.0094±1.0684|

SSIM
|  Fold   | 1 | 2 | 3 | 4 | 5 | avg |
|:----------:|:----:|:----:|:----:|:----:|:----:|:----:|
|bicubic  |0.9779±0.0041|0.9752±0.0042|0.9780±0.0043|0.9786±0.0041|0.9769±0.0060|0.9773±0.0045|
|SRCNN    |0.9858±0.0020|0.9840±0.0022|0.9851±0.0037|0.9853±0.0027|0.9844±0.0042|0.9849±0.0030|
|VDSR     |0.9868±0.0017|0.9862±0.0023|0.9868±0.0035|0.9870±0.0022|0.9869±0.0040|0.9867±0.0027|
|DRRN     |0.9872±0.0019|0.9871±0.0023|0.9873±0.0033|0.9873±0.0019|0.9872±0.0037|0.9872±0.0026|
|StereoSR |0.9869±0.0015|0.9861±0.0023|0.9867±0.0036|0.9870±0.0020|0.9868±0.0037|0.9867±0.0026|
|PASSR    |0.9870±0.0017|0.9861±0.0023|0.9868±0.0035|0.9871±0.0019|0.9870±0.0039|0.9868±0.0027|
|DCSSR    |0.9876±0.0016|0.9871±0.0023|0.9873±0.0034|0.9875±0.0021|0.9873±0.0037|0.9874±0.0026|
|a1       |0.9874±0.0015|0.9870±0.0023|0.9871±0.0034|0.9875±0.0020|0.9872±0.0039|0.9872±0.0026|
|a2       |0.9875±0.0016|0.9871±0.0023|0.9873±0.0033|0.9876±0.0021|0.9873±0.0037|0.9874±0.0026|
|a3       |0.9874±0.0016|0.9870±0.0022|0.9872±0.0034|0.9874±0.0020|0.9872±0.0036|0.9872±0.0026|
|a4       |0.9874±0.0016|0.9870±0.0023|0.9873±0.0034|0.9874±0.0020|0.9872±0.0037|0.9873±0.0026|
