import common from "@/styles/commonStyles";
import { alpha } from "@mui/material";

export default {
    root: {
        position: "relative",
        overflow: "hidden",
    },
    heading: (color) => ({
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: color && alpha(color, 0.2),
    }),
    body: {
        padding: "20px 0px",
    },
    headIcon: {
        margin: "0px 0.7rem",
    },
    btnContainer: {
        display: "flex",
        justifyContent: "center",
    },
    btn: {
        margin: "0px 5px",
    },
    notification: (active) => ({
        display: "flex",
        alignItems: "center",
        position: "absolute",
        bottom: active ? "0px" : "-70px",
        left: "0px",
        backgroundColor: (theme) => theme.palette.background.paper,
        padding: "10px 20px",
        paddingRight: "5px",
        margin: "5px",
        borderRadius: "5px",
        transition: "0.3s",
        zIndex: 1,
        boxShadow: (theme) => theme.shadows[10],
    }),
    cross: {
        ...common.iconBtn,
        margin: "7px",
        padding: "2px",
    },
};
