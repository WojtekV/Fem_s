def n1(ksi, eta):
    return (1/4)*(1-ksi)*(1-eta)


def n2(ksi, eta):
    return (1/4)*(1+ksi)*(1-eta)


def n3(ksi, eta):
    return (1/4)*(1+ksi)*(1+eta)


def n4(ksi, eta):
    return (1/4)*(1-ksi)*(1+eta)


def dn1_dksi(eta):
    return -(1/4)*(1-eta)


def dn2_dksi(eta):
    return (1/4)*(1-eta)


def dn3_dksi(eta):
    return (1/4)*(1+eta)


def dn4_dksi(eta):
    return -(1/4)*(1+eta)


def dn1_deta(ksi):
    return -(1/4)*(1-ksi)


def dn2_deta(ksi):
    return -(1/4)*(1+ksi)


def dn3_deta(ksi):
    return (1/4)*(1+ksi)


def dn4_deta(ksi):
    return (1/4)*(1-ksi)


dn_dksi = [dn1_dksi, dn2_dksi, dn3_dksi, dn4_dksi]
dn_deta = [dn1_deta, dn2_deta, dn3_deta, dn4_deta]
n = [n1, n2, n3, n4]
