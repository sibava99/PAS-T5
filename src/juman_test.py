from pyknp import Jumanpp
jumanpp = Jumanpp()   # default is JUMAN++: Juman(jumanpp=True). if you use JUMAN, use Juman(jumanpp=False)
result = jumanpp.analysis("下 →	鴨 神社の参道は暗かった。")
l = [mrph.midasi for mrph in result.mrph_list()]
print(l)