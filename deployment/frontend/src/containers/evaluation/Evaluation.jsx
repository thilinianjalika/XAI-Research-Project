import Textarea from "@/components/Textarea";
import { Box, Typography } from "@mui/material";
import { LoadingButton } from "@mui/lab";
import { ThumbDownOffAlt, ThumbUpOffAlt } from "@mui/icons-material";
import styles from "./styles";
import commonStyles from "@/styles/commonStyles";
import { useContext, useRef, useState } from "react";
import { getSentiment } from "@/functions/api";
import { ModalContext } from "@/providers/modalProvider/ModalProvider";

const Evaluation = ({ model }) => {
    const { setNotification } = useContext(ModalContext);
    const textareaRef = useRef();
    const [loading, setLoading] = useState(false);
    const [prompt, setPrompt] = useState("");
    const [result, setResult] = useState();
    const evaluationHandler = () => {
        const prompt = textareaRef.current.value;
        if (prompt !== "" && model) {
            setLoading(true);
            setPrompt(prompt);
            setResult();
            getSentiment(model, prompt)
                .then((sentiment) => setResult(sentiment))
                .catch(setNotification)
                .finally(() => {
                    setLoading(false);
                });
        }
    };
    const status = result && result.prediction === "positive";

    return (
        <Box sx={commonStyles.sectionContainer}>
            <Typography variant="h3">Prompt Evaluation</Typography>
            <Textarea
                ref={textareaRef}
                placeholder="Prompt"
                sx={commonStyles.text}
            />
            <LoadingButton
                loading={loading}
                sx={commonStyles.btn}
                onClick={evaluationHandler}
            >
                Evaluate
            </LoadingButton>
            {prompt !== "" && (
                <Box sx={commonStyles.outputContainer(status)}>
                    <Typography>{prompt}</Typography>
                    {status !== undefined && (
                        <Box sx={styles.iconWrapper}>
                            {status ? <ThumbUpOffAlt /> : <ThumbDownOffAlt />}
                            <Typography>{result.score}</Typography>
                        </Box>
                    )}
                </Box>
            )}
        </Box>
    );
};

export default Evaluation;
