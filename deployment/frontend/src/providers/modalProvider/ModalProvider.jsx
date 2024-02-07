import ModalContainer from "@/components/modalContainer/ModalContainer";
import { Box, Button, IconButton, Typography, useTheme } from "@mui/material";
import { createContext, useState } from "react";
import styles from "./styles";
import { Close, ErrorOutline } from "@mui/icons-material";

export const ModalContext = createContext();

const ModalProvider = ({ children }) => {
    const theme = useTheme();
    const [modalContent, setModalContent] = useState();
    const [notificationText, setNotificationText] = useState();
    const [notificationActive, setNotificationActive] = useState(false);

    const displayModal = ({
        heading,
        body,
        state,
        closeBtnText,
        closeBtnProps,
        closeBtnExtraActions,
        extraBtnTexts,
        extraBtnActions,
    }) => {
        let color = undefined;
        let headIcon = undefined;
        if (state === "error") {
            color = theme.palette.error.main;
            headIcon = (
                <ErrorOutline
                    color={"error"}
                    fontSize="inherit"
                    sx={styles.headIcon}
                />
            );
        }
        const extraBtns = [];
        if (extraBtnTexts) {
            for (let i = 0; i < extraBtnTexts.length; i++) {
                const text = extraBtnTexts[i];
                const action = extraBtnActions[i];
                extraBtns.push({ text, action });
            }
        }
        let [sx, rest] = [undefined, undefined];
        if (closeBtnProps) {
            const { sxN, ...restN } = closeBtnProps;
            sx = sxN;
            rest = restN;
        }
        setModalContent({
            headIcon,
            heading,
            body,
            color,
            closeBtnText,
            closeBtnProps: rest,
            closeBtnSx: sx,
            closeBtnExtraActions,
            extraBtns,
        });
    };
    const setNotification = (text) => {
        setNotificationText(text);
        setNotificationActive(true);
        setTimeout(() => {
            setNotificationActive(false);
        }, 3000);
    };
    const closeModal = () => {
        modalContent.closeBtnExtraActions &&
            modalContent.closeBtnExtraActions();
        setModalContent();
    };
    return (
        <ModalContext.Provider
            value={{ displayModal, setNotification, closeModal }}
        >
            <Box sx={styles.root}>
                {children}
                {modalContent && (
                    <ModalContainer show={true} close={closeModal}>
                        <Typography
                            sx={styles.heading(modalContent.color)}
                            variant="h3"
                        >
                            {modalContent.headIcon} {modalContent.heading}
                        </Typography>
                        {typeof modalContent.body === "string" ? (
                            <Typography sx={styles.body} variant="body1">
                                {modalContent.body}
                            </Typography>
                        ) : (
                            modalContent.body
                        )}
                        <Box sx={styles.btnContainer}>
                            {modalContent.extraBtns.map((btn) => (
                                <Button
                                    key={btn.text}
                                    sx={styles.btn}
                                    variant="outlined"
                                    onClick={(e) => {
                                        btn.action(e);
                                        closeModal();
                                    }}
                                >
                                    {btn.text}
                                </Button>
                            ))}
                            <Button
                                sx={{
                                    ...styles.btn,
                                    ...modalContent.closeBtnSx,
                                }}
                                {...modalContent.closeBtnProps}
                                variant="outlined"
                                onClick={closeModal}
                            >
                                {modalContent.closeBtnText
                                    ? modalContent.closeBtnText
                                    : "Close"}
                            </Button>
                        </Box>
                    </ModalContainer>
                )}
                <Box sx={styles.notification(notificationActive)}>
                    <Typography variant="body1">{notificationText}</Typography>
                    <IconButton
                        sx={styles.cross}
                        onClick={() => setNotificationActive(false)}
                    >
                        <Close />
                    </IconButton>
                </Box>
            </Box>
        </ModalContext.Provider>
    );
};

export default ModalProvider;
