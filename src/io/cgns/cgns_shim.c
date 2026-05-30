#include <cgnslib.h>

/* CGNS cg_goto 为变参 API；Rust 侧通过此 shim 读取 BC 关联 Family 名。 */
int asimu_cg_read_boco_family_name(int fn, int B, int Z, int BC, char *family_name) {
    if (cg_goto(fn, B, "Zone_t", Z, "ZoneBC_t", 1, "BC_t", BC, "end") != 0) {
        return 1;
    }
    return cg_famname_read(family_name);
}
