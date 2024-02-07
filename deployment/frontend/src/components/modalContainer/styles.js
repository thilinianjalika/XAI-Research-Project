import commonStyles from "@/styles/commonStyles";
export default {
    modal: {
        minWidth: "640px",
    },
    headRibbon: {
        display: "flex",
        justifyContent: "flex-end",
    },
    cross: {
        ...commonStyles.iconBtn,
        margin: "7px",
        padding: "2px",
    },
    body: {
        padding: "20px",
        paddingTop: "0px",
    },
};
