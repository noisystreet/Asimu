#include <cgnslib.h>
#include <stddef.h>
#include <stdio.h>

/* CGNS cg_goto 为变参 API；Rust 侧通过此 shim 读取 BC 关联 Family 名。 */
int asimu_cg_read_boco_family_name(int fn, int B, int Z, int BC, char *family_name) {
    if (cg_goto(fn, B, "Zone_t", Z, "ZoneBC_t", 1, "BC_t", BC, "end") != 0) {
        return 1;
    }
    return cg_famname_read(family_name);
}

/* 写出结构化网格 + 顶点原始变量（ρ, u, v, w, p @ Vertex，便于 ParaView Volume 显示）。 */
int asimu_cg_write_structured_flow(
    const char *filename,
    const char *basename,
    const char *zonename,
    int nx,
    int ny,
    int nz,
    const double *points_x,
    const double *points_y,
    const double *points_z,
    const double *rho,
    const double *u,
    const double *v,
    const double *w,
    const double *p,
    double physical_time
) {
    int fn = 0;
    int base = 0;
    int zone = 0;
    int sol = 0;
    int coord = 0;
    int field = 0;
    /* CGNS 4.x 结构化 zone：顶点尺寸 + 单元尺寸 + 边界顶点占位（共 9 元素）。 */
    cgsize_t isize[9];
    int err;
    char desc[64];

    err = cg_open(filename, CG_MODE_WRITE, &fn);
    if (err != CG_OK) {
        return err;
    }

    err = cg_base_write(fn, basename, 3, 3, &base);
    if (err != CG_OK) {
        cg_close(fn);
        return err;
    }

    isize[0] = (cgsize_t)(nx + 1);
    isize[1] = (cgsize_t)(ny + 1);
    isize[2] = (cgsize_t)(nz + 1);
    isize[3] = (cgsize_t)nx;
    isize[4] = (cgsize_t)ny;
    isize[5] = (cgsize_t)nz;
    isize[6] = 0;
    isize[7] = 0;
    isize[8] = 0;
    err = cg_zone_write(fn, base, zonename, isize, CGNS_ENUMV(Structured), &zone);
    if (err != CG_OK) {
        cg_close(fn);
        return err;
    }

    err = cg_coord_write(
        fn, base, zone, CGNS_ENUMV(RealDouble), "CoordinateX", points_x, &coord);
    if (err != CG_OK) {
        cg_close(fn);
        return err;
    }
    err = cg_coord_write(
        fn, base, zone, CGNS_ENUMV(RealDouble), "CoordinateY", points_y, &coord);
    if (err != CG_OK) {
        cg_close(fn);
        return err;
    }
    err = cg_coord_write(
        fn, base, zone, CGNS_ENUMV(RealDouble), "CoordinateZ", points_z, &coord);
    if (err != CG_OK) {
        cg_close(fn);
        return err;
    }

    err = cg_sol_write(fn, base, zone, "FlowSolution", CGNS_ENUMV(Vertex), &sol);
    if (err != CG_OK) {
        cg_close(fn);
        return err;
    }

    if (physical_time >= 0.0) {
        snprintf(desc, sizeof(desc), "physical_time=%.16e", physical_time);
        if (cg_goto(fn, base, "Zone_t", zone, "FlowSolution_t", sol, "end") == CG_OK) {
            cg_descriptor_write("PhysicalTime", desc);
        }
    }

    err = cg_field_write(
        fn, base, zone, sol, CGNS_ENUMV(RealDouble), "Density", rho, &field);
    if (err != CG_OK) {
        cg_close(fn);
        return err;
    }
    err = cg_field_write(
        fn, base, zone, sol, CGNS_ENUMV(RealDouble), "VelocityX", u, &field);
    if (err != CG_OK) {
        cg_close(fn);
        return err;
    }
    err = cg_field_write(
        fn, base, zone, sol, CGNS_ENUMV(RealDouble), "VelocityY", v, &field);
    if (err != CG_OK) {
        cg_close(fn);
        return err;
    }
    err = cg_field_write(
        fn, base, zone, sol, CGNS_ENUMV(RealDouble), "VelocityZ", w, &field);
    if (err != CG_OK) {
        cg_close(fn);
        return err;
    }
    err = cg_field_write(
        fn, base, zone, sol, CGNS_ENUMV(RealDouble), "Pressure", p, &field);
    if (err != CG_OK) {
        cg_close(fn);
        return err;
    }

    err = cg_close(fn);
    return err;
}
