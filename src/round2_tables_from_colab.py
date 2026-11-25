"""Tables from Colab steps 10-12 (values copied from results_controls_clip.json, results_morality_robustness.json,
results_coco_translation.json on Drive)."""
import pandas as pd
F = ['lowlevel', 'objects', 'supervised_vit', 'dinov2', 'clip']
probe = {  # rho, lo, hi for valence, morality, arousal, morality_beyond_valence
 'lowlevel': [(.2633,.2282,.2947),(.2283,.197,.2606),(.2251,.1878,.2607),(.1972,.1618,.2303)],
 'objects': [(.6474,.6227,.6726),(.6154,.5898,.6403),(.624,.5992,.648),(.5209,.4943,.5492)],
 'supervised_vit': [(.7006,.6776,.7199),(.6736,.6528,.6945),(.6573,.6338,.6795),(.5764,.5525,.6009)],
 'dinov2': [(.7607,.7421,.7781),(.7353,.7183,.7522),(.7238,.7039,.7417),(.5918,.565,.6169)],
 'clip': [(.8918,.8825,.8994),(.8659,.8543,.8759),(.8253,.8093,.8372),(.697,.6765,.7162)]}
T = ['valence', 'morality', 'arousal', 'morality_beyond_valence']
pd.DataFrame([{'features': f, 'target': t, 'rho': v[0], 'lo': v[1], 'hi': v[2]} for f in F for t, v in zip(T, probe[f])]).to_csv('probes_by_features.csv', index=False)

bc = {  # controls -> target -> (controls_alone, dino rho lo hi, clip rho lo hi)
 'lowlevel': {'valence': (.2633,(.7177,.6988,.7372),(.8622,.8519,.8717)), 'morality': (.2283,(.6985,.6793,.717),(.844,.832,.8546)), 'arousal': (.2251,(.6854,.6666,.7046),(.7921,.7766,.8066))},
 'lowlevel+objects': {'valence': (.6468,(.3823,.3508,.4095),(.6512,.6272,.6691)), 'morality': (.6151,(.352,.3219,.3835),(.6102,.5837,.6348)), 'arousal': (.6233,(.3071,.2766,.3382),(.5466,.52,.5696))},
 'lowlevel+objects+supervised': {'valence': (.6894,(.3562,.3255,.386),(.6487,.6253,.6696)), 'morality': (.6641,(.3213,.2879,.3552),(.6025,.5787,.6278)), 'arousal': (.6505,(.2812,.2466,.309),(.54,.5136,.5671))}}
rows = []
for c, d in bc.items():
    for t, (alone, dn, cl) in d.items():
        for m, v in [('dinov2', dn), ('clip', cl)]:
            rows.append({'controls': c, 'target': t, 'model': m, 'controls_alone': alone, 'rho_on_residual': v[0], 'lo': v[1], 'hi': v[2]})
pd.DataFrame(rows).to_csv('beyond_controls.csv', index=False)

ax = {  # space -> axis -> (valence, morality, arousal, morality|valence, valence|morality)
 'clip': {'affect': (.6689,.4985,-.3291,-.2152,.5466), 'moral': (.5438,.5538,-.2808,.1981,.1547), 'arousal': (.2668,.1897,-.4018,-.0851,.2085)},
 'dinov2_map_fit_on_smid': {'affect': (.6182,.4666,-.2757,-.1732,.4836), 'moral': (.4613,.4922,-.1918,.2091,.0813), 'arousal': (.2242,.132,-.4162,-.1268,.2213)},
 'dinov2_map_fit_on_coco': {'affect': (.5251,.3599,-.2344,-.2212,.4567), 'moral': (.2899,.2805,-.126,.0617,.0982), 'arousal': (.1815,.0311,-.4283,-.2555,.3086)}}
pd.DataFrame([{'space': s, 'axis': a, 'valence': v[0], 'morality': v[1], 'arousal': v[2], 'morality_given_valence': v[3], 'valence_given_morality': v[4]}
              for s, d in ax.items() for a, v in d.items()]).to_csv('language_axes.csv', index=False)

al = {  # space -> direction -> z for affect, moral, arousal axes
 'clip': {'valence': (4.96,4.67,3.46), 'morality': (3.15,4.66,2.31), 'morality_beyond_valence': (-2.27,1.34,-.86), 'arousal': (-.99,-1.71,-4.97)},
 'dinov2_map_fit_on_smid': {'valence': (4.67,3.82,2.45), 'morality': (2.34,4.69,2.01), 'morality_beyond_valence': (-.93,1.34,0.0), 'arousal': (-1.35,-1.72,-3.46)},
 'dinov2_map_fit_on_coco': {'valence': (1.9,1.44,1.69), 'morality': (1.03,1.84,1.08), 'morality_beyond_valence': (-1.12,1.09,-.02), 'arousal': (-.58,-1.62,-1.51)}}
pd.DataFrame([{'space': s, 'direction': d, 'z_affect_axis': v[0], 'z_moral_axis': v[1], 'z_arousal_axis': v[2]} for s, dd in al.items() for d, v in dd.items()]).to_csv('direction_alignment.csv', index=False)

tr = [{'map_fit_on': 'smid (5-fold, held-out images)', 'mean_cosine': .8207, 'mean_cosine_random_pairs': .6051, 'top1': .3636, 'top5': .6677},
      {'map_fit_on': 'coco val2017 (5000 images)', 'mean_cosine': .7709, 'mean_cosine_random_pairs': .587, 'top1': .3575, 'top5': .5799}]
pd.DataFrame(tr).assign(n_candidates=2940, chance_top1=round(1/2940, 5)).to_csv('translation_quality.csv', index=False)

rb = [('linear(valence)', .2432, .189, .5918, .697, .5209, .5764, .1972),
      ('spline(valence)', .233, .1973, .6088, .7023, .5256, .5766, .1805),
      ('spline(valence)+spline(arousal)', .2248, .2045, .5842, .6801, .508, .5537, .1664)]
pd.DataFrame(rb, columns=['residual_method', 'residual_share_of_morality_variance', 'noise_share_of_residual', 'dinov2', 'clip', 'objects', 'supervised_vit', 'lowlevel']).to_csv('morality_beyond_valence_robustness.csv', index=False)

W = {
 'clip': {'valence': (["peace","happiness","joy","a family","kindness","fairness","a desert","money","love","generosity","friendship","a crowd"],
                      ["pain","anger","cruelty","abuse","murder","suffering","war","horror","violence","despair","fear","a cigarette"]),
          'morality_beyond_valence': (["respect","grief","a hospital","care","compassion","a funeral","a nurse","a cemetery","war","a soldier","an elderly person","blood"],
                                      ["a phone","a boat","an airplane","a car","happiness","a bird","a river","a bicycle","a desert","a chair","a blurry image","a horse"])},
 'dinov2_map_fit_on_smid': {'valence': (["love","peace","kindness","a family","happiness","joy","a mountain","a desert","fairness","the color green","friendship","a sunny day"],
                                        ["war","murder","a battlefield","pain","a cigarette","violence","suffering","cruelty","a slum","horror","despair","terrorism"]),
          'morality_beyond_valence': (["respect","a funeral","a nurse","compassion","a soldier","a classroom","care","a cemetery","loyalty","a hospital","an elderly person","bright light"],
                                      ["exploitation","money","a phone","a river","smoke","fun","a factory","a mountain","a boat","a tree","a forest","a prison"])},
 'dinov2_map_fit_on_coco': {'valence': (["fairness","a desert","the color yellow","love","peace","happiness","kindness","the color blue","bright light","joy","a couple","comfort"],
                                        ["murder","pollution","blood","smoke","a slum","vandalism","a funeral","a prison","horror","poverty","a city street","an elderly person"]),
          'morality_beyond_valence': (["respect","the color yellow","care","disgust","justice","a nurse","a funeral","kindness","war","the color green","a cemetery","blood"],
                                      ["a parking lot","a car","rain","smoke","the color blue","a toy","exploitation","a city street","crime","laughter","an aerial view","happiness"])}}
pd.DataFrame([{'space': s, 'direction': d, 'end': end, 'rank': i + 1, 'word': w}
              for s, dd in W.items() for d, (top, bot) in dd.items() for end, ws in [('top', top), ('bottom', bot)] for i, w in enumerate(ws)]).to_csv('direction_words.csv', index=False)
print('ok')
