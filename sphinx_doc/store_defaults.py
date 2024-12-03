from glow import lenses, time_domain, time_domain_c, freq_domain, freq_domain_c

def out_params(mod, cls, froot='source/generated/'):
    cls.__init__ = lambda *args, **kwargs: None

    params = cls().default_general_params()
    name = mod.__name__ + '.' + cls.__name__

    str_dic = lambda root, p: root + ' = ' + (', \n' + (len(root)+3)*' ').join(p.__repr__().split(','))

    try:
        out_phys = str_dic("    p_phys", params[0]) + '\n'
        out_prec = str_dic("    p_prec", params[1])
    except KeyError:
        out_phys = ''
        out_prec = str_dic('    p_prec', params)

    out = '.. code-block:: python\n\n'
    out += out_phys + out_prec

    fname = froot + name + '.txt'

    with open(fname, "w") as fout:
        fout.write(out)

if __name__ == '__main__':
    print(" * Storing lenses defaults", end='')
    out_params(lenses, lenses.Psi_SIS)
    out_params(lenses, lenses.Psi_CIS)
    out_params(lenses, lenses.Psi_PointLens)
    out_params(lenses, lenses.Psi_Ball)
    out_params(lenses, lenses.Psi_NFW)
    out_params(lenses, lenses.Psi_tSIS)
    out_params(lenses, lenses.Psi_eSIS)
    out_params(lenses, lenses.Psi_eCIS)
    out_params(lenses, lenses.Psi_Ext)
    out_params(lenses, lenses.Psi_offcenterSIS)
    out_params(lenses, lenses.Psi_offcenterCIS)
    out_params(lenses, lenses.Psi_offcenterPointLens)
    out_params(lenses, lenses.Psi_offcenterBall)
    out_params(lenses, lenses.Psi_offcenterNFW)
    print(" -> Done")

    print(" * Storing time_domain defaults", end='')
    out_params(time_domain, time_domain.It_SingleContour)
    out_params(time_domain, time_domain.It_SingleIntegral)
    out_params(time_domain, time_domain.It_AnalyticSIS)
    out_params(time_domain, time_domain.It_NaiveAreaIntegral)
    print(" -> Done")

    print(" * Storing time_domain_c defaults", end='')
    out_params(time_domain, time_domain_c.It_SingleContour_C)
    out_params(time_domain, time_domain_c.It_SingleIntegral_C)
    out_params(time_domain, time_domain_c.It_AnalyticSIS_C)
    out_params(time_domain, time_domain_c.It_AreaIntegral_C)
    out_params(time_domain, time_domain_c.It_MultiContour_C)
    print(" -> Done")

    print(" * Storing freq_domain defaults", end='')
    out_params(freq_domain, freq_domain.Fw_FFT_OldReg)
    out_params(freq_domain, freq_domain.Fw_SemiAnalyticSIS)
    out_params(freq_domain, freq_domain.Fw_AnalyticPointLens)
    print(" -> Done")

    print(" * Storing freq_domain_c defaults", end='')
    out_params(freq_domain, freq_domain_c.Fw_FFT_C)
    out_params(freq_domain, freq_domain_c.Fw_SemiAnalyticSIS_C)
    out_params(freq_domain, freq_domain_c.Fw_AnalyticPointLens_C)
    out_params(freq_domain, freq_domain_c.Fw_DirectFT_C)
    print(" -> Done")

