    # ---- 034 NEGPROXY: NEGATIVE-ONLY molecules (no positive spectrum at all) get their [M-H]- spectra as
    # [M+H]+ proxies (prec and every peak + 2*1.007276, relabelled positive). Train panel, 40 mols x 16 same-formula
    # cands: ICE MRR 0.681 / GL 0.731 vs random 0.227 (real [M+H]+ control 0.852 / 0.883). Any failure -> unchanged.
    NEG_PROXY, n_negp = True, 0
    if NEG_PROXY:
        try:
            _all = {it['mid']: it for it in ice_fuse.build_ice_input(te, cands, covered_only=False)}
            for it in items:
                if it['spectra'] or any(sp.get('mode') == 'positive' for sp in _all.get(it['mid'], {}).get('spectra', [])):
                    continue             # only NEGATIVE-ONLY molecules (keeps N1 NH4/WL proxies and its ICE complement intact)
                for s in _all.get(it['mid'], {}).get('spectra', []):
                    if s.get('adduct') == '[M-H]-' and s.get('mz'):
                        it['spectra'].append(dict(s, mz=[x + 2.014552 for x in s['mz']], prec=s['prec'] + 2.014552,
                                                  adduct='[M+H]+', mode='positive'))
                        n_negp += 1
            print('034 NEGPROXY spectra', n_negp, 'molecules', sum(1 for it in items if any(
                sp.get('mode') == 'positive' for sp in it['spectra'])), flush=True)
        except Exception as _e:
            print('034 NEGPROXY skipped', repr(_e), flush=True)
