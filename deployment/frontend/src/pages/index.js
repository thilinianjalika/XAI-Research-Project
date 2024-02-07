import Head from "next/head";
import { Box } from "@mui/material";
import pageStyles from "@/styles/pageStyles/index";
import Evaluation from "@/containers/evaluation/Evaluation";
import { useState } from "react";
import Stats from "@/containers/stats/Stats";
import ModelSelection from "@/containers/modelSelection/ModelSelection";
import Analysis from "@/containers/analysis/Analysis";
import { MODEL_NAME_LR } from "@/constants";

export default function Home() {
    const [model, setModel] = useState();
    return (
        <>
            <Head>
                <title>XAI</title>
                <meta
                    name="description"
                    content="An implementation of explainable AI algorithms"
                />
                <meta
                    name="viewport"
                    content="width=device-width, initial-scale=1"
                />
                <link rel="icon" href="/favicon.ico" />
            </Head>
            <Box sx={pageStyles.main}>
                <ModelSelection setModel={setModel} />

                {model && (
                    <>
                        <Evaluation model={model} />
                        <Stats model={model} />
                        <Analysis model={model} />
                    </>
                )}
            </Box>
        </>
    );
}
